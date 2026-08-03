"""High-level API for the ayah alignment pipeline.

    from ayah_aligner import AyahAligner

    aligner = AyahAligner()                       # loads models ONCE
    result = aligner.align(audio_path, surah=16, ayahs=range(1, 19))
    result.save(out_dir)                          # writes manifest.json + .wav + figure per ayah

    for entry in result:
        print(entry.ayah, entry.confident, entry.start_s, entry.end_s)

Or, for one-shot use:

    from ayah_aligner import crop_ayahs
    result = crop_ayahs(audio_path, surah=16, ayahs=range(1, 19), out_dir=out_dir)

This is a thin, reusable wrapper around the actual algorithm, which lives in
boundary_refine.py (resolve_ayah, segment_run_candidates, windowed_similarity,
get_segments, finalize_start/end) and is unchanged from crop_pipeline.py --
only the packaging is new. See boundary_refine.resolve_ayah()'s docstring for
the reasoning behind each design choice; in short:

  - Ayahs are processed in NUMBER order (not confidence order), since most
    reciters go sequentially most of the time.
  - Each ayah tries STAGED anchor attempts, never pooled: (1) the immediately
    preceding CONFIRMED ayah's end, then (2) a global chunk-similarity
    search, which needs a STRICTER score to confirm since a global search
    can land on a region that plausibly matches several different ayahs'
    text at once.
  - Candidates are real-pause segment-runs first, a fixed-window fallback
    only when no real pause exists nearby.
  - Length is chosen by scoring every plausible run length up front plus a
    duration-over-estimate penalty -- there is deliberately no post-hoc
    "extend if uncertain" step (three variants of one were tried and each
    ran away into a multi-ayah span at least once).
  - Anything that clears neither attempt is skipped, not forced, then
    retried in a RESIDUAL PASS once every other ayah is confirmed: real
    unclaimed time gaps are computed and every skipped ayah is scored
    against real segments specifically inside them.
  - Every result carries an explicit `confident` flag and `warnings` --
    nothing is silently presented as correct. Ear verification against the
    saved crop is still the real ground truth this pipeline serves; it
    narrows a whole recording down to a short, trustworthy list of
    candidates to check, not a replacement for checking them.

Validated end-to-end on one 18-ayah, non-sequential recitation: 18/18
matching a fully hand-verified, ear-checked ground truth. See this module's
`align()` docstring for what is and isn't expected to generalize as-is to a
different recording, reciter, or embedding model.
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
import boundary_refine as br
SR = 16000
DEFAULT_TEXT_MODEL_NAME = 'jhu-clsp/mmBERT-base'
DEFAULT_QURAN_API = 'https://quranapi.pages.dev/api/{surah}.json'
DEFAULT_AUDIO_MODEL_PATH = 'FaisaI/tadabur-embedding'
DEFAULT_CKPT_PATH = 'auto'
CKPT_REPO = 'FaisaI/tadabur-embedding'
CKPT_FILENAME = 'checkpoint_last.pt'
PAIRS_JSONL_REPO = 'FaisaI/tadabur-align-references'
PAIRS_JSONL_FILENAME = 'aligner_pairs.jsonl'

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

    @property
    def duration_s(self):
        return None if self.start_s is None else round(self.end_s - self.start_s, 3)

class AlignmentResult:

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
        for e in self.entries:
            if not e.confident:
                continue
            filename = f'S{self.surah}_A{e.ayah:03d}.wav'
            s, en = (e.start_s, e.end_s)
            sf.write(os.path.join(out_dir, filename), self._wave[0, int(s * self._sr):int(en * self._sr)].numpy(), self._sr)
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

class AyahAligner:

    def __init__(self, audio_model_path=DEFAULT_AUDIO_MODEL_PATH, ckpt_path=DEFAULT_CKPT_PATH, text_model_name=DEFAULT_TEXT_MODEL_NAME, quality_floor=0.45, fallback_floor=0.75, default_rate=0.2, min_dur=0.4, verbose=True, device=None, segmenter_device=None):
        from transformers import AutoModel, AutoTokenizer
        self.quality_floor = quality_floor
        self.fallback_floor = fallback_floor
        self.default_rate = default_rate
        self.min_dur = min_dur
        self.verbose = verbose
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
                raise RuntimeError('could not auto-download the alignment checkpoint -- pass ckpt_path= explicitly to AyahAligner() with a local .pt file instead')
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

    def align(self, audio_path, surah, ayahs=None, ayah_start=None, ayah_end=None, canon_text=None, pace_hints=None, pairs_jsonl='auto', quran_api=DEFAULT_QURAN_API):
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
        claimed_mask = np.zeros(len(chunk_centers), dtype=bool)
        raw_results = {}
        skipped = []

        def confirmed_spans():
            return [(r['start'], r['end']) for r in raw_results.values() if r['confident']]
        for ayah in ayahs:
            p = prep[ayah]
            text, query, exp_dur, exp_dur_raw = (p['text'], p['query'], p['exp_dur'], p['exp_dur_raw'])
            anchor_groups = []
            prev = raw_results.get(ayah - 1)
            if prev and prev['confident']:
                anchor_groups.append([prev['end']])
            anchor_groups.append(top_peaks(query, 3, claimed_mask, min_sep=max(3.0, exp_dur)))
            other_queries = [prep[a]['query'] for a in ayahs if a != ayah]
            result = br.resolve_ayah(text, query, wave, total, self._embed_text, embed_clips, anchor_groups, exp_dur, exp_dur_raw, confirmed_spans(), quality_floor=self.quality_floor, fallback_floor=self.fallback_floor, other_queries=other_queries)
            s, e = self._snap_to_silence(wave, total, br.finalize_start(wave, total, result['start'], result['end']), result['end'])
            e = br.finalize_end(wave, total, s, e)
            s, e = self._snap_to_silence(wave, total, s, e)
            result['start'], result['end'] = (s, e)
            raw_results[ayah] = result
            if result['confident']:
                claimed_mask |= (chunk_centers >= s - 0.5) & (chunk_centers <= e + 0.5)
            else:
                skipped.append(ayah)
            self._log(f"  ayah {ayah:2d}: [{s:6.2f},{e:6.2f}] ({e - s:5.2f}s) {result['method']:14s} sc={result['sc']:.3f} open={result['open_score']:.3f} close={result['close_score']:.3f}  {('CONFIDENT' if result['confident'] else 'SKIPPED: ' + ','.join(result['warnings']))}")

        def _accept_residual(ayah, s0, e0, sc, warning):
            s, e = self._snap_to_silence(wave, total, br.finalize_start(wave, total, s0, e0), e0)
            e = br.finalize_end(wave, total, s, e)
            s, e = self._snap_to_silence(wave, total, s, e)
            o, c = br.edge_match_scores(self._embed_text, embed_clips, prep[ayah]['text'], s, e)
            raw_results[ayah] = {'method': 'residual_gap', 'start': s, 'end': e, 'sc': sc, 'open_score': o, 'close_score': c, 'confident': True, 'warnings': [warning]}
            self._log(f'  RESOLVED ayah {ayah} at [{s:.2f},{e:.2f}] sc={sc:.3f} ({warning})')
        if skipped:
            gaps = br.find_gaps(confirmed_spans(), total)
            self._log(f'  residual pass: {len(skipped)} skipped, {len(gaps)} unclaimed gaps')
            all_best = {}
            for glo, ghi in gaps:
                segs = br.get_segments(wave, total, glo, ghi, pad=8.0)
                candidates_per_ayah = {ayah: [] for ayah in skipped}
                for i in range(len(segs)):
                    for j in range(i, min(i + 4, len(segs))):
                        a, b = (segs[i][0], segs[j][1])
                        if b - a > 25:
                            break
                        for ayah in skipped:
                            sc = br.windowed_similarity(embed_clips, a, b, prep[ayah]['query'])
                            candidates_per_ayah[ayah].append((sc, a, b))
                best_per_ayah = {}
                for ayah, cands in candidates_per_ayah.items():
                    if not cands:
                        continue
                    top = max((c[0] for c in cands))
                    close = [c for c in cands if c[0] >= top - 0.03]
                    sc, a, b = min(close, key=lambda c: c[2] - c[1])
                    best_per_ayah[ayah] = (sc, a, b)
                for ayah, (sc, a, b) in best_per_ayah.items():
                    if ayah not in all_best or sc > all_best[ayah][0]:
                        all_best[ayah] = (sc, a, b)
                    if sc > 0.3 and (not br.overlaps_any(a, b, confirmed_spans())):
                        _accept_residual(ayah, a, b, sc, 'found_via_residual_gap_pass')
                        skipped.remove(ayah)
            if len(skipped) == 1:
                ayah = skipped[0]
                if ayah in all_best:
                    sc, a, b = all_best[ayah]
                    if not br.overlaps_any(a, b, confirmed_spans()):
                        _accept_residual(ayah, a, b, sc, 'resolved_by_elimination_last_remaining_ayah')
                        skipped.remove(ayah)
        entries = []
        for ayah in ayahs:
            r = raw_results.get(ayah)
            if r is None:
                entries.append(AyahResult(ayah, prep[ayah]['text'], None, None, False, 'none', 0.0, 0.0, 0.0, ['not_processed']))
            else:
                entries.append(AyahResult(ayah, prep[ayah]['text'], round(r['start'], 3), round(r['end'], 3), r['confident'], r['method'], round(r['sc'], 4), round(r['open_score'], 4), round(r['close_score'], 4), r['warnings']))
        n_conf = sum((1 for e in entries if e.confident))
        self._log(f'done: {n_conf}/{len(ayahs)} confident  (total {time.time() - t0:.1f}s)')
        if skipped:
            self._log(f'needs manual attention: {skipped}')
        return AlignmentResult(entries, wave, SR, surah, audio_path)

def crop_ayahs(audio_path, surah, ayahs=None, ayah_start=None, ayah_end=None, out_dir=None, aligner=None, **aligner_kwargs):
    aligner = aligner or AyahAligner(**aligner_kwargs)
    result = aligner.align(audio_path, surah, ayahs=ayahs, ayah_start=ayah_start, ayah_end=ayah_end)
    if out_dir:
        result.save(out_dir)
    return result