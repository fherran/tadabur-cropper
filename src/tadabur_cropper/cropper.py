"""Crop ayahs out of a recitation recording: audio-text embedding search,
plus reference-based boundary refinement and mutashabihat verification.

    from tadabur_cropper import AyahCropper
    result = AyahCropper().align("rec.mp3", surah=26, ayah_start=160, ayah_end=163)
    result.save("out/")
"""
import json
import os
import ssl
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable
import certifi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from . import boundary_refine as br
SR = 16000
DEFAULT_TEXT_MODEL_NAME = 'jhu-clsp/mmBERT-base'
DEFAULT_QURAN_API = 'https://quranapi.pages.dev/api/{surah}.json'
DEFAULT_AUDIO_MODEL_PATH = 'FaisaI/tadabur-embedding'
DEFAULT_CKPT_PATH = 'auto'
CKPT_REPO = 'FaisaI/tadabur-embedding'
CKPT_FILENAME = 'checkpoint_last.pt'
PAIRS_JSONL_REPO = 'FaisaI/tadabur-align-references'
PAIRS_JSONL_FILENAME = 'aligner_pairs.jsonl'
MUTASHABIHAT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mutashabihat', 'mutashabihat.json')
_mutashabihat_cache = None

def _load_mutashabihat():
    global _mutashabihat_cache
    if _mutashabihat_cache is None:
        try:
            with open(MUTASHABIHAT_PATH, encoding='utf-8') as f:
                _mutashabihat_cache = json.load(f)
        except OSError:
            _mutashabihat_cache = {}
    return _mutashabihat_cache

def _hf_download(repo_id, filename, repo_type, what):
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    except Exception as e:
        print(f'{what} auto-download failed ({e})', flush=True)
        return None

def _download_default_pairs_jsonl():
    path = _hf_download(PAIRS_JSONL_REPO, PAIRS_JSONL_FILENAME, 'dataset', 'pace-hints')
    if path is None:
        print('falling back to flat default rate', flush=True)
    return path

def _word_overlap(text_a, text_b):
    """Shared words / shorter text's word count (bag overlap)."""
    wa, wb = text_a.split(), text_b.split()
    if not wa or not wb:
        return 0.0
    remaining, shared = list(wb), 0
    for w in wa:
        if w in remaining:
            remaining.remove(w)
            shared += 1
    return shared / min(len(wa), len(wb))


def _auto_pace_hints(pairs_jsonl, surah, ayahs):
    by_ayah = {}
    try:
        with open(pairs_jsonl, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                if d['surah_id'] == surah - 1 and d['ayah_id'] in ayahs:
                    by_ayah.setdefault(d['ayah_id'], []).append(d)
    except OSError:
        return {}
    hints = {}
    for ayah, rows in by_ayah.items():
        rates = sorted(((d['end_ms'] - d['start_ms']) / 1000 / len(d['text'].replace(' ', '')) for d in rows if len(d['text'].replace(' ', '')) >= 10))
        if rates:
            hints[ayah] = rates[len(rates) // 2]
    return hints

@dataclass
class AyahResult:
    ayah: int
    text: str
    start_s: Optional[float]
    end_s: Optional[float]
    confident: bool
    method: str
    score: float
    open_score: float
    close_score: float
    warnings: list = field(default_factory=list)
    audio_filename: Optional[str] = None
    refine_mad_ms: Optional[float] = None

    @property
    def duration_s(self):
        return None if self.start_s is None else round(self.end_s - self.start_s, 3)

class CropResult:

    def __init__(self, entries, wave, sr, surah, source_path):
        self.entries = entries
        self._wave = wave
        self._sr = sr
        self.surah = surah
        self.source_path = source_path

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, ayah_number):
        for e in self.entries:
            if e.ayah == ayah_number:
                return e
        raise KeyError(ayah_number)

    @property
    def confident_count(self):
        return sum((1 for e in self.entries if e.confident))

    @property
    def skipped(self):
        return [e.ayah for e in self.entries if not e.confident]

    def to_manifest(self):
        out = []
        for e in self.entries:
            d = asdict(e)
            d['duration_s'] = e.duration_s
            out.append(d)
        return out

    def save(self, out_dir, with_figures=True):
        os.makedirs(out_dir, exist_ok=True)
        if with_figures:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        total = self._wave.shape[1] / self._sr
        # slice crops from the original file, not the 16kHz working copy
        src_wav, src_sr = sf.read(self.source_path, dtype='float32', always_2d=True)
        for e in self.entries:
            if not e.confident:
                continue
            filename = f'S{self.surah}_A{e.ayah:03d}.wav'
            s, en = (e.start_s, e.end_s)
            crop = src_wav[int(s * src_sr):int(en * src_sr)]
            sf.write(os.path.join(out_dir, filename), crop, src_sr)
            e.audio_filename = filename
            if with_figures:
                fig, ax = plt.subplots(figsize=(11, 3.2))
                lo, hi = (max(0.0, s - 4.0), min(total, en + 4.0))
                seg = self._wave[0, int(lo * self._sr):int(hi * self._sr)].numpy()
                t = np.arange(len(seg)) / self._sr + lo
                ax.plot(t[::4], seg[::4], lw=0.35, color='#555')
                ax.axvspan(s, en, color='#2ca02c', alpha=0.25)
                ax.axvline(s, color='#1a7a1a', ls='--', lw=1.2)
                ax.axvline(en, color='#1a7a1a', ls='--', lw=1.2)
                ax.set_xlim(lo, hi)
                ax.set_title(f'Surah {self.surah}:{e.ayah}  [{s:.2f}s-{en:.2f}s] ({en - s:.2f}s) {e.method} sc={e.score:.2f}', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'S{self.surah}_A{e.ayah:03d}_figure.png'), dpi=100)
                plt.close(fig)
        with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(self.to_manifest(), f, ensure_ascii=False, indent=2)
        return out_dir

class AyahCropper:

    def __init__(self, audio_model_path=DEFAULT_AUDIO_MODEL_PATH, ckpt_path=DEFAULT_CKPT_PATH, text_model_name=DEFAULT_TEXT_MODEL_NAME, quality_floor=0.45, fallback_floor=0.75, default_rate=0.2, min_dur=0.4, verbose=True, device=None, segmenter_device=None):
        from transformers import AutoModel, AutoTokenizer
        self.quality_floor = quality_floor
        self.fallback_floor = fallback_floor
        self.default_rate = default_rate
        self.min_dur = min_dur
        self.verbose = verbose
        self._refiner = None
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        br.set_device(segmenter_device if segmenter_device is not None else self.device)
        self.audio_model = AutoModel.from_pretrained(audio_model_path, trust_remote_code=True).eval()
        if ckpt_path == 'auto':
            ckpt_path = _hf_download(CKPT_REPO, CKPT_FILENAME, 'model', 'checkpoint')
            if ckpt_path is None:
                raise RuntimeError('could not auto-download the alignment checkpoint -- pass ckpt_path= explicitly to AyahCropper() with a local .pt file instead')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        class ProjectionHead(nn.Module):

            def __init__(self, i, o):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o))

            def forward(self, x):
                return F.normalize(self.net(x), dim=-1)
        self.text_proj = ProjectionHead(768, 384)
        self.text_proj.load_state_dict(ckpt['text_proj_semantic'])
        self.text_proj.eval()
        renames = {'modality_encoders.IMAGE.local_encoder.proj.weight': 'model.local_encoder.proj.weight', 'modality_encoders.IMAGE.local_encoder.proj.bias': 'model.local_encoder.proj.bias', 'modality_encoders.IMAGE.extra_tokens': 'model.extra_tokens', 'modality_encoders.IMAGE.fixed_positional_encoder.positions': 'model.fixed_positional_encoder.positions', 'modality_encoders.IMAGE.context_encoder.norm.weight': 'model.pre_norm.weight', 'modality_encoders.IMAGE.context_encoder.norm.bias': 'model.pre_norm.bias'}
        sd = {renames.get(k, 'model.' + k): v for k, v in ckpt['audio_encoder'].items()}
        sd.update({'model.proj_semantic.' + k: v for k, v in ckpt['audio_proj_semantic'].items()})
        sd.update({'model.proj_speaker.' + k: v for k, v in ckpt['audio_proj_speaker'].items()})
        self.audio_model.load_state_dict(sd, strict=False)
        self.audio_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name).eval()
        self.audio_model.to(self.device)
        self.text_proj.to(self.device)
        self.text_model.to(self.device)
        self._log(f'models loaded (device={self.device})')

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    @torch.no_grad()
    def _embed_text(self, text):
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.text_model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-06)
        return self.text_proj(pooled)[0].cpu()

    def _embed_clips(self, wave, spans):
        mels = []
        for s, e in spans:
            seg = wave[:, int(s * SR):int(e * SR)]
            seg = seg - seg.mean()
            m = torchaudio.compliance.kaldi.fbank(seg, htk_compat=True, sample_frequency=SR, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            m = F.pad(m, (0, 0, 0, 1024 - m.shape[0])) if m.shape[0] < 1024 else m[:1024]
            mels.append((m - -4.381) / (3.628 * 2))
        with torch.no_grad():
            batch = torch.stack(mels)[:, None].to(self.device)
            return self.audio_model.semantic_embedding(batch).cpu()

    def _snap_to_silence(self, wave, total, s, e, max_ext=1.5):
        hop = 0.025
        env = wave[0][:int(total / hop) * int(hop * SR)].reshape(-1, int(hop * SR)).pow(2).mean(1).sqrt()
        quiet = (env < 0.04 * env.max()).numpy()

        def nudge(t, direction):
            i = int(t / hop)
            for step in range(int(max_ext / hop)):
                j = i + direction * step
                if 0 <= j < len(quiet) and quiet[j]:
                    return j * hop
            return t
        return (max(0.0, nudge(s, -1)), min(total, nudge(e, +1)))

    def _windowed_scores(self, embed_clips, s, e, queries):
        """windowed_similarity scoring for many queries, one embedding pass."""
        if e - s <= 10.2:
            embs = embed_clips([(s, e)])
            return [float((embs @ q)[0]) for q in queries]
        starts = list(np.arange(s, e - 10.0 + 1e-06, 5.0)) or [s]
        if starts[-1] + 10.0 < e:
            starts.append(e - 10.0)
        embs = embed_clips([(a, min(e, a + 10.0)) for a in starts])
        return [float((embs @ q).max()) for q in queries]

    def align(self, audio_path, surah, ayahs=None, ayah_start=None, ayah_end=None, canon_text=None, pace_hints=None, pairs_jsonl='auto', quran_api=DEFAULT_QURAN_API, residual_max_span=60.0, residual_max_run=10, mask_fatiha=True, decoy_margin=0.35, refine=True):
        t0 = time.time()
        if canon_text is None:
            ctx = ssl.create_default_context(cafile=certifi.where())
            req = urllib.request.Request(quran_api.format(surah=surah), headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                canon_text = json.loads(r.read().decode())['arabic2']
        if ayahs is not None:
            ayahs = list(ayahs)
        else:
            start = ayah_start if ayah_start is not None else 1
            end = ayah_end if ayah_end is not None else len(canon_text)
            ayahs = list(range(start, end + 1))
            self._log(f"no explicit ayah list -- using {('ayah_start/ayah_end' if ayah_start or ayah_end else 'the whole surah')}: {start}-{end} ({len(ayahs)} ayahs)")
        if pace_hints is None:
            if pairs_jsonl == 'auto':
                pairs_jsonl = _download_default_pairs_jsonl()
            pace_hints = _auto_pace_hints(pairs_jsonl, surah, ayahs) if pairs_jsonl else {}
            self._log(f"pace hints: {len(pace_hints)}/{len(ayahs)} ayahs calibrated from dataset{('' if pairs_jsonl else ' (skipped, pairs_jsonl unavailable)')}")
        wav, sr = sf.read(audio_path, dtype='float32')
        wave = torch.from_numpy(wav.T if wav.ndim > 1 else wav[None]).mean(0, keepdim=True)
        if sr != SR:
            wave = torchaudio.functional.resample(wave, sr, SR)
        total = wave.shape[1] / SR
        embed_clips = lambda spans: self._embed_clips(wave, spans)
        chunk_w, chunk_hop = (3.0, 1.0)
        chunk_spans = [(float(s), float(min(s + chunk_w, total))) for s in np.arange(0, max(total - chunk_hop, 0) + 1e-06, chunk_hop)]
        chunk_embs = embed_clips(chunk_spans)
        chunk_centers = np.array([(s + e) / 2 for s, e in chunk_spans])
        self._log(f'[{time.time() - t0:.0f}s] {os.path.basename(audio_path)}: {total:.1f}s indexed as {len(chunk_spans)} chunks')

        def text_for(ayah):
            try:
                return canon_text[ayah - 1]
            except (IndexError, TypeError):
                return canon_text[ayah]

        def top_peaks(query, k, claimed_mask, min_sep):
            sims = (chunk_embs @ query).numpy()
            sims = np.where(claimed_mask, -np.inf, sims)
            order = np.argsort(-sims)
            peaks = []
            for oi in order:
                if sims[oi] == -np.inf:
                    break
                t = float(chunk_centers[oi])
                if all((abs(t - p) > min_sep for p in peaks)):
                    peaks.append(t)
                if len(peaks) >= k:
                    break
            return peaks
        prep = {}
        for ayah in ayahs:
            text = text_for(ayah)
            full_chars = len(text.replace(' ', ''))
            rate = pace_hints.get(ayah, self.default_rate)
            exp_dur_raw = max(full_chars * rate, self.min_dur + 0.2)
            exp_dur = min(exp_dur_raw, 10.24)
            query = self._embed_text(text)
            prep[ayah] = {'text': text, 'exp_dur': exp_dur, 'exp_dur_raw': exp_dur_raw, 'query': query}
        # in-request rivals, logged only; never arbitrate by embedding score
        # (tried, noise) -- the word duel in refine is the real check
        confusable = {ayah: set() for ayah in ayahs}
        for i, a1 in enumerate(ayahs):
            for a2 in ayahs[i + 1:]:
                if _word_overlap(prep[a1]['text'], prep[a2]['text']) >= 0.8:
                    confusable[a1].add(a2)
                    confusable[a2].add(a1)
        n_confusable = sum(1 for v in confusable.values() if v)
        if n_confusable:
            self._log(f'  {n_confusable}/{len(ayahs)} ayahs have a textually confusable rival in this request')
        # prayer recordings recite al-Fatiha mid-file; reject spans matching
        # it decisively better than the requested ayah
        decoy_queries = []
        if mask_fatiha and surah != 1:
            try:
                ctx = ssl.create_default_context(cafile=certifi.where())
                req = urllib.request.Request(quran_api.format(surah=1), headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                    fatiha_texts = json.loads(r.read().decode())['arabic2']
                decoy_queries = [self._embed_text(t) for t in fatiha_texts]
                self._log(f'  fatiha decoys: {len(decoy_queries)} ayahs embedded (spans matching al-Fatiha decisively better than the requested ayah are rejected)')
            except Exception as ex:
                self._log(f'  fatiha decoys unavailable ({ex}) -- proceeding without')

        def decoy_check(s, e, own_query):
            """(hit, own_sc, best_decoy_sc); decoys also scored on ~3s sub-windows."""
            if not decoy_queries:
                return (False, 0.0, 0.0)
            own = self._windowed_scores(embed_clips, s, e, [own_query])[0]
            sub = [(s, e)] if e - s <= 3.5 else []
            t = s
            while t + 2.0 < e:
                sub.append((t, min(e, t + 3.0)))
                t += 1.5
            embs = embed_clips(sub)
            best_decoy = max(float((embs @ dq).max()) for dq in decoy_queries)
            return (best_decoy - own > decoy_margin, own, best_decoy)
        claimed_mask = np.zeros(len(chunk_centers), dtype=bool)
        raw_results = {}
        skipped = []
        # actual/predicted duration ratios: this reciter's observed pace
        pace_ratios = []

        def confirmed_spans():
            return [(r['start'], r['end']) for r in raw_results.values() if r['confident']]

        def order_bounds(ayah):
            """Valid time window for this ayah given confirmed numeric neighbors."""
            lower = 0.0
            for a in range(ayah - 1, min(ayahs) - 1, -1):
                r = raw_results.get(a)
                if r and r['confident']:
                    lower = r['end']
                    break
            upper = total
            for a in range(ayah + 1, max(ayahs) + 1):
                r = raw_results.get(a)
                if r and r['confident']:
                    upper = r['start']
                    break
            return (lower, upper)
        for ayah in ayahs:
            p = prep[ayah]
            text, query, exp_dur, exp_dur_raw = (p['text'], p['query'], p['exp_dur'], p['exp_dur_raw'])
            anchor_groups = []
            prev = raw_results.get(ayah - 1)
            if prev and prev['confident']:
                anchor_groups.append([prev['end']])
            anchor_groups.append(top_peaks(query, 3, claimed_mask, min_sep=max(3.0, exp_dur)))
            # decoys stay OUT of other_queries (segment-level margins are noise);
            # finished spans only
            other_queries = [prep[a]['query'] for a in ayahs if a != ayah]
            result = br.resolve_ayah(text, query, wave, total, self._embed_text, embed_clips, anchor_groups, exp_dur, exp_dur_raw, confirmed_spans(), quality_floor=self.quality_floor, fallback_floor=self.fallback_floor, other_queries=other_queries)
            s, e = self._snap_to_silence(wave, total, br.finalize_start(wave, total, result['start'], result['end']), result['end'])
            e = br.finalize_end(wave, total, s, e)
            s, e = self._snap_to_silence(wave, total, s, e)
            result['start'], result['end'] = (s, e)
            if result['confident'] and decoy_queries:
                hit, own_sc, best_decoy = decoy_check(s, e, query)
                if hit:
                    result['confident'] = False
                    result['warnings'] = result['warnings'] + ['span_matches_fatiha_decoy_better']
                    self._log(f'  ayah {ayah}: REJECTED -- span [{s:.2f},{e:.2f}] matches al-Fatiha decisively better ({best_decoy:.3f} vs own {own_sc:.3f}); likely the prayer\'s Fatiha, not this ayah')
            raw_results[ayah] = result
            if result['confident']:
                claimed_mask |= (chunk_centers >= s - 0.5) & (chunk_centers <= e + 0.5)
                pace_ratios.append((e - s) / exp_dur_raw)
            else:
                skipped.append(ayah)
            self._log(f"  ayah {ayah:2d}: [{s:6.2f},{e:6.2f}] ({e - s:5.2f}s) {result['method']:14s} sc={result['sc']:.3f} open={result['open_score']:.3f} close={result['close_score']:.3f}  {('CONFIDENT' if result['confident'] else 'SKIPPED: ' + ','.join(result['warnings']))}")

        # demote spans that contradict the Quran's fixed ayah order;
        # looped to a fixpoint since one demotion can expose another
        changed, rounds = True, 0
        while changed and rounds < 5:
            changed, rounds = False, rounds + 1
            for ayah in sorted(ayahs):
                r = raw_results[ayah]
                if not r['confident']:
                    continue
                lower, upper = order_bounds(ayah)
                if r['start'] < lower - 0.5 or r['end'] > upper + 0.5:
                    self._log(f'  ayah {ayah}: DEMOTED -- [{r["start"]:.2f},{r["end"]:.2f}] contradicts ayah sequence '
                               f'order (must fall within [{lower:.2f},{upper:.2f}] given other confirmed ayahs); deferring to residual pass')
                    r['confident'] = False
                    r['warnings'] = r['warnings'] + ['demoted_violates_ayah_sequence_order']
                    skipped.append(ayah)
                    changed = True

        def _accept_residual(ayah, s0, e0, sc, warning):
            s, e = self._snap_to_silence(wave, total, br.finalize_start(wave, total, s0, e0), e0)
            e = br.finalize_end(wave, total, s, e)
            s, e = self._snap_to_silence(wave, total, s, e)
            o, c = br.edge_match_scores(self._embed_text, embed_clips, prep[ayah]['text'], s, e)
            # residual matches are global-search class: confident only above
            # fallback_floor, else kept as a flagged guess
            prev = raw_results.get(ayah)
            if prev and prev.get('method') == 'residual_gap' and prev['sc'] >= sc:
                return prev['confident']
            warnings = [warning]
            decoy_hit, _, _ = decoy_check(s, e, prep[ayah]['query'])
            if decoy_hit:
                warnings.append('span_matches_fatiha_decoy_better')
            confident = sc >= self.fallback_floor and not decoy_hit
            if not confident and not decoy_hit:
                warnings.append('below_fallback_floor_verify_by_ear')
            raw_results[ayah] = {'method': 'residual_gap', 'start': s, 'end': e, 'sc': sc, 'open_score': o, 'close_score': c, 'confident': confident, 'warnings': warnings}
            self._log(f'  {"RESOLVED" if confident else "GUESSED (unverified)"} ayah {ayah} at [{s:.2f},{e:.2f}] sc={sc:.3f} ({",".join(warnings)})')
            return confident
        if skipped:
            observed_pace = float(np.median(pace_ratios)) if pace_ratios else 1.0
            self._log(f'  residual pass: observed pace {observed_pace:.2f}x pace_hints prediction (from {len(pace_ratios)} confirmed ayahs so far)')
            gaps = br.find_gaps(confirmed_spans(), total)
            self._log(f'  residual pass: {len(skipped)} skipped, {len(gaps)} unclaimed gaps')
            all_best = {}
            for glo, ghi in gaps:
                segs = br.get_segments(wave, total, glo, ghi, pad=8.0)
                candidates_per_ayah = {ayah: [] for ayah in skipped}
                # caps are backstops; the real bound is the pace-scaled soft penalty
                for i in range(len(segs)):
                    for j in range(i, min(i + residual_max_run, len(segs))):
                        a, b = (segs[i][0], segs[j][1])
                        if b - a > residual_max_span:
                            break
                        for ayah in skipped:
                            # never score a span outside the ayah's order window
                            lower, upper = order_bounds(ayah)
                            if a < lower - 0.5 or b > upper + 0.5:
                                continue
                            sc = br.windowed_similarity(embed_clips, a, b, prep[ayah]['query'])
                            candidates_per_ayah[ayah].append((sc, a, b))
                best_per_ayah = {}
                for ayah, cands in candidates_per_ayah.items():
                    if not cands:
                        continue
                    ref_dur = prep[ayah]['exp_dur_raw'] * observed_pace

                    def penalized(c, ref_dur=ref_dur):
                        sc, a, b = c
                        over = max(0.0, b - a - 1.4 * ref_dur)
                        return sc - over / max(ref_dur, 3.0) * 0.6
                    top = max((penalized(c) for c in cands))
                    close = [c for c in cands if penalized(c) >= top - 0.03]
                    non_overlapping = [c for c in close if not br.overlaps_any(c[1], c[2], confirmed_spans())]
                    sc, a, b = max(non_overlapping or close, key=lambda c: c[2] - c[1])
                    best_per_ayah[ayah] = (sc, a, b)
                for ayah, (sc, a, b) in best_per_ayah.items():
                    if ayah not in all_best or sc > all_best[ayah][0]:
                        all_best[ayah] = (sc, a, b)
                    if sc > 0.3 and (not br.overlaps_any(a, b, confirmed_spans())):
                        if _accept_residual(ayah, a, b, sc, 'found_via_residual_gap_pass'):
                            skipped.remove(ayah)
            if len(skipped) == 1:
                ayah = skipped[0]
                if ayah in all_best:
                    sc, a, b = all_best[ayah]
                    if not br.overlaps_any(a, b, confirmed_spans()):
                        if _accept_residual(ayah, a, b, sc, 'resolved_by_elimination_last_remaining_ayah'):
                            skipped.remove(ayah)
        # Boundary refinement: word timestamps transferred from reference
        # recitations (tadabur-align) replace loudness-based edges.
        if refine:
            api = None
            try:
                from tadabur_align import WordTimestamps
                from tadabur_align import verify as ta_verify
                if self._refiner is None:
                    self._refiner = WordTimestamps(device=self.device)
                api = self._refiner
            except Exception as ex:
                self._log(f'  refine skipped ({ex})')
            DUEL_MARGIN = 0.08

            if api is not None:
                for ayah in ayahs:
                    r = raw_results.get(ayah)
                    if not (r and r['confident']):
                        continue
                    try:
                        lower, upper = order_bounds(ayah)
                        prev_r, next_r = raw_results.get(ayah - 1), raw_results.get(ayah + 1)
                        if prev_r and prev_r['end'] <= r['start'] + 0.1:
                            lower = max(lower, prev_r['end'])
                        if next_r and next_r['start'] >= r['end'] - 0.1:
                            upper = min(upper, next_r['start'])
                        lo = max(0.0, lower, r['start'] - 0.8)
                        hi = min(total, max(upper, r['end']), r['end'] + 0.8)
                        med, mad, tc2, tf2, own_refs = ta_verify.word_stamps(api, wave[:, int(lo * SR):int(hi * SR)], surah, ayah)
                        new_s, new_e = lo + med[0, 0], lo + med[-1, 1]
                        moved = []
                        r['_edge'] = (max(lo, new_s - 0.06) if mad[0, 0] <= 150 else None,
                                      min(hi, new_e + 0.1) if mad[-1, 1] <= 150 else None)
                        # shared boundaries are set by the joint pass below, not here
                        start_shared = prev_r and prev_r['confident'] and 0 <= r['start'] - prev_r['end'] <= 2.0
                        end_shared = next_r and next_r['confident'] and 0 <= next_r['start'] - r['end'] <= 2.0
                        if not start_shared and mad[0, 0] <= 150 and abs(new_s - r['start']) > 0.08:
                            r['start'] = max(lo, new_s - 0.06)
                            moved.append('start')
                        if not end_shared and mad[-1, 1] <= 150 and abs(new_e - r['end']) > 0.08:
                            r['end'] = min(hi, new_e + 0.1)
                            moved.append('end')
                        if moved:
                            r['start'], r['end'] = self._snap_to_silence(wave, total, r['start'], r['end'])
                            self._log(f'  ayah {ayah}: refined {"+".join(moved)} -> [{r["start"]:.2f},{r["end"]:.2f}] (word agreement {mad.mean():.0f}ms)')
                        r['refine_mad_ms'] = round(float(mad.mean()), 1)
                        if mad.mean() > 250:
                            r['warnings'] = r['warnings'] + ['weak_word_agreement_verify_by_ear']
                        verdicts = ta_verify.word_duel(api, med, tc2, tf2, own_refs, _load_mutashabihat().get(f'{surah}:{ayah}', {}))
                        if verdicts:
                            worst = min(verdicts, key=lambda v: v[2] - v[1])
                            rk, own_c, riv_c = worst
                            if riv_c + DUEL_MARGIN < own_c:
                                r['confident'] = False
                                r['warnings'] = r['warnings'] + [f'differing_word_matches_rival_{rk}']
                                self._log(f'  ayah {ayah}: REJECTED by word duel -- audio matches ayah {rk} better ({riv_c:.3f} vs own {own_c:.3f})')
                            elif own_c + DUEL_MARGIN < riv_c:
                                self._log(f'  ayah {ayah}: word duel verified vs {len(verdicts)} rival(s) (own {own_c:.3f} vs best rival {riv_c:.3f})')
                            else:
                                r['warnings'] = r['warnings'] + ['differing_word_duel_inconclusive_verify_by_ear']
                                self._log(f'  ayah {ayah}: word duel inconclusive vs {rk} ({riv_c:.3f} vs own {own_c:.3f}) -- verify by ear')
                    except Exception as ex:
                        r['warnings'] = r['warnings'] + ['refine_failed']
                        self._log(f'  ayah {ayah}: refine failed ({ex})')

                # recovery first: (B) a weak-agreement neighbor probably
                # swallowed the skipped ayah -> stitch the pair and split;
                # then (A) fit remaining skipped ayahs into free gaps, best
                # word-agreement per gap wins
                for _round in range(3):
                    progressed = False
                    for ayah in ayahs:
                        r = raw_results.get(ayah)
                        prev_r = raw_results.get(ayah - 1)
                        if not (r and not r['confident'] and prev_r and prev_r['confident'] and prev_r.get('refine_mad_ms', 0) > 250):
                            continue
                        try:
                            lo2, hi2 = prev_r['start'], min(total, prev_r['end'] + 0.3)
                            med, mad, splits, smad = ta_verify.stitched_stamps(api, wave[:, int(lo2 * SR):int(hi2 * SR)], surah, [ayah - 1, ayah])
                            split = lo2 + splits[0]
                            if smad[0] <= 300 and lo2 + 0.5 < split < hi2 - 0.5:
                                old_end = prev_r['end']
                                prev_r['end'] = split
                                prev_r['refine_mad_ms'] = round(float(mad.mean()), 1)
                                prev_r['warnings'] = prev_r['warnings'] + [f'shrunk_gave_tail_to_{ayah}']
                                e2 = min(hi2, lo2 + med[-1, 1] + 0.1)
                                raw_results[ayah] = {'method': 'recovered_neighbor_split', 'start': split, 'end': e2, 'sc': r.get('sc', 0.0), 'open_score': 0.0, 'close_score': 0.0, 'confident': True, 'warnings': ['recovered_from_neighbor_split_verify_by_ear'], 'refine_mad_ms': round(float(mad.mean()), 1), '_end_capped': e2 >= hi2 - 0.12}
                                self._log(f'  ayah {ayah}: RECOVERED from inside ayah {ayah - 1} -- split at {split:.2f}s (neighbor end was {old_end:.2f})')
                                progressed = True
                        except Exception as ex:
                            self._log(f'  ayah {ayah}: neighbor-split recovery failed ({ex})')
                    windows = {}
                    for ayah in ayahs:
                        r = raw_results.get(ayah)
                        if r and not r['confident']:
                            lower, upper = order_bounds(ayah)
                            if upper - lower >= 1.0:
                                windows.setdefault((round(lower, 2), round(upper, 2)), []).append(ayah)
                    for (wl, wu), cands in windows.items():
                        best = None
                        for ayah in cands:
                            try:
                                lo2, hi2 = max(0.0, wl), min(total, wu)
                                med, mad, tc2, tf2, own_refs = ta_verify.word_stamps(api, wave[:, int(lo2 * SR):int(hi2 * SR)], surah, ayah)
                                if mad.mean() <= 250 and (best is None or mad.mean() < best[1]):
                                    best = (ayah, float(mad.mean()), med, lo2, hi2)
                            except Exception:
                                pass
                        if best is not None:
                            ayah, m, med, lo2, hi2 = best
                            s2 = max(lo2, lo2 + med[0, 0] - 0.06)
                            e2 = min(hi2, lo2 + med[-1, 1] + 0.1)
                            raw_results[ayah] = {'method': 'recovered_gap_words', 'start': s2, 'end': e2, 'sc': raw_results[ayah].get('sc', 0.0), 'open_score': 0.0, 'close_score': 0.0, 'confident': True, 'warnings': ['recovered_word_alignment_verify_by_ear'], 'refine_mad_ms': round(m, 1), '_start_capped': s2 <= lo2 + 0.08, '_end_capped': e2 >= hi2 - 0.12}
                            self._log(f'  ayah {ayah}: RECOVERED in gap [{s2:.2f},{e2:.2f}] (word agreement {m:.0f}ms, {len(cands)} candidate(s))')
                            progressed = True
                    if not progressed:
                        break

                # then stitch each chain of adjacent confident ayahs in ONE
                # alignment; uncertain chain splits fall back to the per-ayah
                # word grids from phase 1
                chains, cur = [], []
                for ayah in ayahs:
                    r = raw_results.get(ayah)
                    if r and r['confident'] and cur and ayah == cur[-1] + 1 and -0.2 <= r['start'] - raw_results[cur[-1]]['end'] <= 2.0:
                        cur.append(ayah)
                    elif r and r['confident']:
                        if len(cur) > 1:
                            chains.append(cur)
                        cur = [ayah]
                    else:
                        if len(cur) > 1:
                            chains.append(cur)
                        cur = []
                if len(cur) > 1:
                    chains.append(cur)
                for chain in chains:
                    try:
                        r_first, r_last = raw_results[chain[0]], raw_results[chain[-1]]
                        lower, _ = order_bounds(chain[0])
                        _, upper = order_bounds(chain[-1])
                        lo = max(0.0, lower, r_first['start'] - 0.8)
                        hi = min(total, max(upper, r_last['end']), r_last['end'] + 0.8)
                        med, mad, splits, smad = ta_verify.stitched_stamps(api, wave[:, int(lo * SR):int(hi * SR)], surah, chain)
                        prev_n = raw_results.get(chain[0] - 1)
                        next_n = raw_results.get(chain[-1] + 1)
                        if mad[0, 0] <= 150:
                            s_new = max(lo, lo + med[0, 0] - 0.06)
                            if prev_n:
                                s_new = max(s_new, min(r_first['start'], prev_n['end']))
                            r_first['start'] = s_new
                        if mad[-1, 1] <= 150:
                            e_new = min(hi, lo + med[-1, 1] + 0.1)
                            if next_n:
                                e_new = min(e_new, max(r_last['end'], next_n['start']))
                            r_last['end'] = e_new
                        for k, sp in enumerate(splits):
                            aa, bb = chain[k], chain[k + 1]
                            ra, rb = raw_results[aa], raw_results[bb]
                            pa = (ra.get('_edge') or (None, None))[1]
                            pb = (rb.get('_edge') or (None, None))[0]
                            cands_ = []
                            if smad[k] <= 150:
                                cands_.append((lo + sp, 'chain'))
                            else:
                                try:
                                    lo_p, hi_p = ra['start'], rb['end']
                                    _, _, sp_p, smad_p = ta_verify.stitched_stamps(api, wave[:, int(lo_p * SR):int(hi_p * SR)], surah, [aa, bb])
                                    if smad_p[0] <= 150:
                                        cands_.append((lo_p + sp_p[0], 'pair'))
                                except Exception:
                                    pass
                            if pa is not None and pb is not None and abs(pa - pb) <= 0.3:
                                cands_.append(((pa + pb) / 2, 'per-ayah'))
                            elif pb is not None:
                                cands_.append((pb, 'per-ayah start'))
                            elif pa is not None:
                                cands_.append((pa, 'per-ayah end'))
                            # a real pause between the pair (narrow gap) is strong
                            # evidence: the split may not wander far from it
                            gap0, gap1 = ra['end'], rb['start']
                            tight = gap1 - gap0 < 1.0 and not (ra.get('_end_capped') or rb.get('_start_capped'))
                            split, src_ = None, ''
                            for cand, name in cands_:
                                if tight and not gap0 - 0.3 <= cand <= gap1 + 0.3:
                                    continue
                                split, src_ = cand, name
                                break
                            if split is not None and ra['start'] + 0.5 < split < rb['end'] - 0.5:
                                old_b = (ra['end'], rb['start'])
                                ra['end'] = split
                                rb['start'] = split
                                self._log(f'  boundary {aa}|{bb}: split at {split:.2f}s via {src_} (was {old_b[0]:.2f}|{old_b[1]:.2f}, chain spread {smad[k]:.0f}ms)')
                            else:
                                ra['warnings'] = ra['warnings'] + ['shared_boundary_unrefined_verify_by_ear']
                    except Exception as ex:
                        self._log(f'  chain {chain[0]}-{chain[-1]}: joint refine failed ({ex})')

        entries = []
        for ayah in ayahs:
            r = raw_results.get(ayah)
            if r is None:
                entries.append(AyahResult(ayah, prep[ayah]['text'], None, None, False, 'none', 0.0, 0.0, 0.0, ['not_processed']))
            else:
                entries.append(AyahResult(ayah, prep[ayah]['text'], round(r['start'], 3), round(r['end'], 3), r['confident'], r['method'], round(r['sc'], 4), round(r['open_score'], 4), round(r['close_score'], 4), r['warnings'], refine_mad_ms=r.get('refine_mad_ms')))
        n_conf = sum((1 for e in entries if e.confident))
        self._log(f'done: {n_conf}/{len(ayahs)} confident  (total {time.time() - t0:.1f}s)')
        if skipped:
            self._log(f'needs manual attention: {skipped}')
        return CropResult(entries, wave, SR, surah, audio_path)

def crop_ayahs(audio_path, surah, ayahs=None, ayah_start=None, ayah_end=None, out_dir=None, aligner=None, **aligner_kwargs):
    aligner = aligner or AyahCropper(**aligner_kwargs)
    result = aligner.align(audio_path, surah, ayahs=ayahs, ayah_start=ayah_start, ayah_end=ayah_end)
    if out_dir:
        result.save(out_dir)
    return result