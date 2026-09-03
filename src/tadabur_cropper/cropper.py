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
MAX_SEQ_FANOUT = 6
SEGMENTER_REPO = 'obadx/recitation-segmenter-v2'
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
        self._segmenter = None
        self._rstack = {}
        self._stops_cache = {}
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

    def _align_by_stops(self, audio_path, wave, total, surah, ayahs, prep, api, pad=0.30, span_score=None):
        """Main path. The reciter marks the boundaries himself by stopping, so
        take his stops as the only candidate cuts and ask, at each one, how far
        into the current ayah we have got. When its last word is reached the
        ayah is done. Nothing here has to guess where a cut belongs."""
        stops = self._merge_false_stops(wave, total, self._segment_stops(audio_path))
        if len(stops) < 2:
            self._log(f'  only {len(stops)} speech span(s) -- nothing to walk')
            return None
        # an ayah with too few clean reference alignments cannot be scored,
        # but it must not sink the whole run: it takes one span and is flagged
        wc, ref_of, unscorable = {}, {}, set()
        for a in ayahs:
            try:
                ref_of[a] = api._references(surah, a)
                wc[a] = len(next(iter(ref_of[a].values()))['words'])
            except Exception as ex:
                unscorable.add(a)
                self._log(f'  ayah {a}: no usable references ({ex}) -- will take one span, flagged')
        if len(unscorable) > len(ayahs) // 3:
            return None

        fcache, ccache = {}, {}
        def feats(lo, hi):
            key = (round(lo, 2), round(hi, 2))
            if key not in fcache:
                fcache[key] = self._feats(api, wave, lo, hi)
            return fcache[key]

        def rcost(lo, hi, a, j, k):
            key = (round(lo, 2), round(hi, 2), a, j, k)
            if key not in ccache:
                ccache[key] = self._range_cost(api, feats(lo, hi), ref_of[a], j, k)
            return ccache[key]

        # which stop begins the first requested ayah? Anything before it
        # (isti'adha, basmala, takbir) simply scores badly for its words.
        a0 = ayahs[0]
        # WHERE does the passage begin? A prayer recording opens with
        # al-Fatiha, and the first requested ayah may be a single short word
        # (36:1 is يس) that scores about the same against anything. So anchor on
        # the most distinctive of the first few ayahs -- the one with the most
        # words -- shortlist the spans where IT may sit (one short word can
        # match anywhere, so no single span is trusted), then try the spans
        # just before each and keep the start that reads the opening ayahs best.
        head = [a for a in ayahs if a not in unscorable][:3]
        if not head:
            return None
        anchor = max(head, key=lambda a: wc[a])
        scan = [i for i in range(len(stops)) if stops[i][1] - stops[i][0] >= 0.8]
        # a coarse read is enough to shortlist: a few word counts per span,
        # not every one -- the probe below reads the shortlist exactly
        ks = sorted({max(1, round(wc[anchor] * f)) for f in (0.25, 0.5, 0.75, 1.0)})
        a_cost = [(min(rcost(*stops[i], anchor, 1, k) for k in ks), i) for i in scan]
        a_cost.sort()

        readable = [a for a in ayahs if a not in unscorable]

        def probe(i0):
            """mean cost of reading the passage from span i0, a few spans deep"""
            tot, cnt, ptr, ai, i = 0.0, 0, 1, 0, i0
            while ai < len(readable) and i < len(stops) and cnt < 8:
                a = readable[ai]
                lo, hi = stops[i]
                if hi - lo < 0.3:
                    i += 1
                    continue
                c, k = min(((rcost(lo, hi, a, ptr, k), k) for k in range(ptr, wc[a] + 1)), key=lambda t: t[0])
                tot += c
                cnt += 1
                if k >= wc[a]:
                    ai += 1
                    ptr = 1
                else:
                    ptr = k + 1
                i += 1
            # compared on equal footing: a start so late that the recording
            # runs out averages fewer readings, and cannot compete with one
            # that reads the full budget
            if ai >= len(readable):
                cnt = 8   # read the whole passage: a full budget's worth
            return (-cnt, tot / max(cnt, 1)) if ai > 0 else (0, 1e9)

        tried, best = set(), None
        for _, j in a_cost[:5]:
            for i0 in range(max(0, j - 4), j + 1):
                if i0 not in tried:
                    tried.add(i0)
                    pc = probe(i0)
                    if best is None or pc < best[0]:
                        best = (pc, i0)
        start_i = best[1]
        self._log('  start candidates: ' + ', '.join(f'{stops[i][0]:.1f}s={c[1]:.3f}' for c, i in
                                                     sorted((probe(i0), i0) for i0 in tried)[:6]))

        def refdur(a, j, k):
            """how long words j..k of ayah a take the reference reciters"""
            ds = [e['words'][k - 1][1] - e['words'][j - 1][0] for e in ref_of[a].values()]
            return float(np.median(ds)) if ds else 1.0

        # results[ayah] = [start, end, cost, starts_at_span_edge, ends_at_span_edge]
        def walk(start_i):
            """read the passage from this span; returns (results, flags, cost per second)"""
            results, flags, ai, ptr, seg_i, csum, dsum = {}, {}, 0, 1, start_i, 0.0, 0.0
            while ai < len(ayahs) and seg_i < len(stops):
                ayah = ayahs[ai]
                lo, hi = stops[seg_i]
                if ayah in unscorable:
                    results[ayah] = [lo, hi, 1.0, True, True]
                    self._log(f'  ayah {ayah}: took the span [{lo:.2f},{hi:.2f}] unscored (no references)')
                    seg_i, ai, ptr = seg_i + 1, ai + 1, 1
                    continue
                if hi - lo < 0.25:
                    seg_i += 1
                    continue
                self._head_repeat = None
                segs = self._read_span(lo, hi, ai, ptr, ayahs, wc, ref_of, unscorable, rcost, refdur, wave, total)
                if self._head_repeat:
                    flags[self._head_repeat[0]] = f'may_open_with_repeat_of_ayah_{self._head_repeat[1]}_verify_by_ear'
                for a_idx, s0, e0, j0, k, kind in segs:
                    a = ayahs[a_idx]
                    if kind == 'repeat':
                        # the reciter said these words again: they stay with their ayah
                        if a in results:
                            results[a][1], results[a][4] = e0, e0 >= hi - 1e-6
                        flags[a] = 'repeated_phrase_verify_by_ear'
                        self._log(f'    [{s0:.2f},{e0:.2f}] repeats the end of ayah {a} -- kept with it')
                        continue
                    n = wc[a]
                    if a not in results:
                        results[a] = [s0, e0, 0.0, s0 <= lo + 1e-6, e0 >= hi - 1e-6]
                    else:
                        results[a][1], results[a][4] = e0, e0 >= hi - 1e-6
                    c = rcost(s0, e0, a, j0, k)
                    results[a][2] = max(results[a][2], c)
                    csum, dsum = csum + c * (e0 - s0), dsum + (e0 - s0)
                    self._log(f'  ayah {a}: [{s0:.2f},{e0:.2f}] holds words {j0}..{k} of {n}'
                              + ('' if k >= n else ' -- continues in the next span'))
                last = [g for g in segs if g[5] != 'repeat'][-1]
                if last[4] >= wc[ayahs[last[0]]]:
                    ai, ptr = last[0] + 1, 1
                else:
                    ai, ptr = last[0], last[4] + 1
                seg_i += 1

            return results, flags, csum / max(dsum, 1e-6)

        # when two starts read the opening almost equally well, the whole
        # passage decides: a wrong start pays for it further on
        close = [i0 for i0 in tried if probe(i0)[0] == best[0][0] and probe(i0)[1] <= 1.15 * best[0][1]]
        walks = {i0: walk(i0) for i0 in sorted(set(close) | {start_i}, key=probe)[:4]}
        # the start that places the most ayahs wins, cost breaking ties; and
        # if the passage does not fit after the chosen start, that start was
        # too late, so earlier candidates are read in full as well
        def choose():
            return max(walks, key=lambda i: (len(walks[i][0]), -walks[i][2]))
        start_i = choose()
        if len(walks[start_i][0]) < len(ayahs):
            for i0 in [i for i in sorted(tried, key=probe) if i < start_i and i not in walks][:3]:
                walks[i0] = walk(i0)
            start_i = choose()
        if len(walks) > 1:
            self._log('  full read from each candidate start: ' + ', '.join(
                f'{stops[i][0]:.1f}s: {len(w[0])} ayahs at {w[2]:.3f}' for i, w in walks.items()))
        results, flags, _ = walks[start_i]
        self._log(f'  passage starts at the span at {stops[start_i][0]:.2f}s '
                  f'({start_i} earlier span(s) are not part of it)')
        # a cut inside a span sits on the nearest energy dip, which may be a
        # few tenths of a second from the true seam; aligning the two ayahs
        # as one piece places it, and is trusted when the references agree
        from tadabur_align import verify as _verify
        for a, b in zip(ayahs, ayahs[1:]):
            if a not in results or b not in results or results[a][4] or a in unscorable or b in unscorable:
                continue
            if abs(results[a][1] - results[b][0]) > 1e-6:
                continue
            lo_, hi_ = results[a][0], results[b][1]
            try:
                _, _, splits, smad = _verify.stitched_stamps(api, wave[:, int(lo_ * SR):int(hi_ * SR)], surah, [a, b])
            except Exception:
                continue
            j = lo_ + splits[0]
            if smad[0] <= 150 and abs(j - results[a][1]) <= 1.0:
                self._log(f'    {a}|{b}: cut moved {results[a][1]:.2f} -> {j:.2f} by the pair alignment '
                          f'(references agree within {smad[0]:.0f}ms)')
                results[a][1] = results[b][0] = j

        # crop edges. Start: a short lead-in before the segmenter's onset,
        # never more than part of the gap to the previous span, since a soft
        # first consonant sits below what the segmenter hears. End: the last
        # word can fade for a second after the voice seems to stop, so an end
        # at a span edge follows the audio down to sustained silence, never
        # less than the usual padding and never into the next span.
        quiet, hop = self._quiet_mask(wave, total)
        span_lo = {round(a, 2): i for i, (a, b) in enumerate(stops)}
        span_hi = {round(b, 2): i for i, (a, b) in enumerate(stops)}
        for a, r in results.items():
            if r[3] and round(r[0], 2) in span_lo:
                i = span_lo[round(r[0], 2)]
                gap = r[0] - (stops[i - 1][1] if i > 0 else 0.0)
                r[0] = max(0.0, r[0] - min(0.25, 0.4 * gap))
            if r[4] and round(r[1], 2) in span_hi:
                i = span_hi[round(r[1], 2)]
                limit = min(r[1] + 1.5, stops[i + 1][0] - 0.05 if i + 1 < len(stops) else total, total)
                f0, f1 = int(r[1] / hop), int(limit / hop)
                fade = next((t * hop for t in range(f0, f1 - 3) if quiet[t:t + 4].all()), limit)
                r[1] = min(limit, max(r[1] + pad, fade))
        if len(results) < max(1, len(ayahs) // 2):
            self._log(f'  stop-based pass only completed {len(results)}/{len(ayahs)} ayahs -- falling back')
            return None

        entries = []
        for ayah in ayahs:
            if ayah not in results:
                entries.append(AyahResult(ayah, prep[ayah]['text'], None, None, False, 'none', 0.0, 0.0, 0.0,
                                          ['not_found_in_stops']))
                continue
            st, en, cost = results[ayah][:3]
            warns = ['no_reference_verify_by_ear'] if ayah in unscorable else (
                [] if cost <= 0.65 else ['weak_word_match_verify_by_ear'])
            if ayah in flags:
                warns.append(flags[ayah])
            entries.append(AyahResult(ayah, prep[ayah]['text'], round(st, 3), round(en, 3), True,
                                      'stops', round(float(cost), 4), 0.0, 0.0, warns))
        self._log(f'done: {len(results)}/{len(ayahs)} by the stops the reciter made')
        return CropResult(entries, wave, SR, surah, audio_path)

    def _merge_false_stops(self, wave, total, stops):
        """A stop the segmenter reports is real only if the audio is actually
        quiet in the gap. A sustained final vowel can come back as two spans
        with a 'pause' between them that the energy never shows."""
        quiet, hop = self._quiet_mask(wave, total)
        out = []
        for lo, hi in stops:
            if out and not quiet[int(out[-1][1] / hop):int(lo / hop) + 1].any():
                self._log(f'  no silence in the gap {out[-1][1]:.2f}-{lo:.2f}: not a stop, spans merged')
                out[-1] = (out[-1][0], hi)
            else:
                out.append((lo, hi))
        return out

    def _energy_dips(self, wave, total, lo, hi, hop=0.025, prominence=0.06, min_gap=0.6):
        """Prominent energy minima inside [lo, hi]: where the reciter softened
        without stopping outright. The segmenter reports full stops; these are
        the quieter transitions it does not mark."""
        try:
            from scipy.signal import find_peaks
        except Exception:
            return []
        env = wave[0][:int(total / hop) * int(hop * SR)].reshape(-1, int(hop * SR)).pow(2).mean(1).sqrt().numpy()
        sm = np.convolve(env, np.ones(5) / 5, mode='same')
        i0, i1 = int((lo + 0.4) / hop), int((hi - 0.4) / hop)
        if i1 - i0 < 8:
            return []
        pk, props = find_peaks(-sm[i0:i1], prominence=prominence, distance=int(min_gap / hop))
        top = sorted(zip(props['prominences'], pk), reverse=True)[:8]
        return sorted((i0 + int(k)) * hop for _, k in top)

    def _read_span(self, lo, hi, ai, ptr, ayahs, wc, ref_of, unscorable, rcost, refdur, wave, total):
        """Faisal's consume loop at dip level. A span is a run of pieces cut
        at the reciter's energy dips; it may hold the rest of one ayah, or
        several whole ayahs and the start of the next. Every way of laying the coming ayahs
        over the pieces is scored -- each piece must be the words it claims
        to be, and words that are not there or left out both cost -- and the
        cheapest reading wins. 'One ayah, no cut' is simply the one-piece
        reading, so a cut is made only where it explains the audio better.
        Returns [(ayah_index, start, end, first_word, last_word, kind)]."""
        cuts = [t for t in self._energy_dips(wave, total, lo, hi) if lo + 0.3 < t < hi - 0.3]
        P = [lo] + sorted(cuts) + [hi]
        np_ = len(P) - 1
        cand = []
        for m in range(4):
            if ai + m >= len(ayahs) or ayahs[ai + m] in unscorable:
                break
            cand.append(ai + m)

        def plausible(d, a, j, k):
            r = refdur(a, j, k)
            return 0.3 * r <= d <= 3.5 * r + 0.5

        INF = float('inf')
        best = {(0, 0): (0.0, [])}   # (piece, ayahs completed) -> (sum cost*dur, segments)
        for j in range(np_):
            for m in range(len(cand)):
                if (j, m) not in best:
                    continue
                base, path = best[(j, m)]
                a = ayahs[cand[m]]
                j0, n = (ptr if m == 0 else 1), wc[a]
                for j2 in range(j + 1, np_ + 1):
                    d = P[j2] - P[j]
                    if not plausible(d, a, j0, n):
                        continue
                    c = base + rcost(P[j], P[j2], a, j0, n) * d
                    seg = path + [(cand[m], P[j], P[j2], j0, n, 'full')]
                    if c < best.get((j2, m + 1), (INF, None))[0]:
                        best[(j2, m + 1)] = (c, seg)
        final = (INF, None)
        for m in range(1, len(cand) + 1):
            if (np_, m) in best and best[(np_, m)][0] < final[0]:
                final = best[(np_, m)]
        # the last ayah may be unfinished at the span end
        for m in range(len(cand)):
            a = ayahs[cand[m]]
            j0, n = (ptr if m == 0 else 1), wc[a]
            for j in range(np_):
                if (j, m) not in best:
                    continue
                base, path = best[(j, m)]
                d = P[np_] - P[j]
                ks = sorted({j0 + round((n - 1 - j0) * f) for f in (0.0, 0.25, 0.5, 0.75)} & set(range(j0, n)))
                if not ks:
                    continue
                kc, kb = min((rcost(P[j], P[np_], a, j0, k), k) for k in ks)
                for k in range(max(j0, kb - 2), min(n - 1, kb + 2) + 1):
                    c = rcost(P[j], P[np_], a, j0, k)
                    if c < kc:
                        kc, kb = c, k
                if not (j == 0 and m == 0) and not plausible(d, a, j0, kb):
                    continue
                c = base + kc * d
                if c < final[0]:
                    final = (c, path + [(cand[m], P[j], P[np_], j0, kb, 'partial')])
        if final[1] is None:
            a = ayahs[cand[0]]
            kc, kb = min((rcost(lo, hi, a, ptr, k), k) for k in range(ptr, wc[a] + 1))
            return [(cand[0], lo, hi, ptr, kb, 'partial' if kb < wc[a] else 'full')]
        if len(final[1]) > 1:
            self._log(f'  span [{lo:.2f},{hi:.2f}] reads as {len(final[1])} pieces '
                      f'(per-second cost {final[0] / (hi - lo):.3f})')
        # a reciter sometimes says the previous ayah's last words again before
        # going on. Reading that as a rule misfired on murattal reciters, so it
        # is only pointed out: the first piece sounds more like the previous
        # ayah's ending than like this ayah's words
        if ptr == 1 and ai > 0 and ayahs[ai - 1] not in unscorable and len(P) > 2:
            pa = ayahs[ai - 1]
            a0 = final[1][0][0]
            n0 = wc[ayahs[a0]]
            for e_head in P[1:min(len(P) - 1, 5)]:
                own = min(rcost(lo, e_head, ayahs[a0], 1, k) for k in range(1, min(n0, 3) + 1))
                prev = rcost(lo, e_head, pa, max(1, wc[pa] - 2), wc[pa])
                if prev < 0.8 * own:
                    self._log(f'    [{lo:.2f},{e_head:.2f}] sounds like the end of ayah {pa} again '
                              f'({prev:.3f} vs {own:.3f} as ayah {ayahs[a0]}) -- flagged')
                    self._head_repeat = (ayahs[a0], pa)
                    break
        return final[1]

    def _segment_stops(self, audio_path):
        """Where the reciter actually stops, from the recitation segmenter
        (obadx/recitation-segmenter-v2). Returns [(start, end)] speech spans."""
        if self._stops_cache.get(audio_path) is not None:
            return self._stops_cache[audio_path]
        try:
            from recitations_segmenter import segment_recitations, clean_speech_intervals
            from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
            if self._segmenter is None:
                proc = AutoFeatureExtractor.from_pretrained(SEGMENTER_REPO)
                mdl = AutoModelForAudioFrameClassification.from_pretrained(SEGMENTER_REPO)
                dt = torch.bfloat16 if self.device == 'cuda' else torch.float32
                self._segmenter = (proc, mdl.to(self.device, dtype=dt), dt)
            proc, mdl, dt = self._segmenter
            x, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            w = torch.from_numpy(x.T).mean(0, keepdim=True)
            if sr != SR:
                w = torchaudio.functional.resample(w, sr, SR)
            out = segment_recitations([w.squeeze(0)], mdl, proc, device=torch.device(self.device), dtype=dt, batch_size=8)[0]
            clean = clean_speech_intervals(out.speech_intervals, out.is_complete,
                                           min_silence_duration_ms=30, min_speech_duration_ms=30,
                                           pad_duration_ms=30, return_seconds=True)
            stops = [(float(a), float(b)) for a, b in clean.clean_speech_intervals]
            self._log(f'  recitation segmenter: {len(stops)} speech spans between stops')
        except Exception as ex:
            self._log(f'  recitation segmenter unavailable ({ex}) -- falling back to the search path')
            stops = []
        self._stops_cache[audio_path] = stops
        return stops

    def _feats(self, api, wave, lo, hi):
        """Target features of [lo, hi], context-stacked once for scoring."""
        from tadabur_align import dtw as _dtw
        tc, tf, _ = api._target_features(wave[:, int(lo * SR):int(hi * SR)])
        return tc, tf, _dtw.prepare(tf)

    def _range_costs(self, api, feats, refs, j, k):
        """Per-reference DTW cost of the claim 'this audio holds words j..k'."""
        from tadabur_align import dtw as _dtw
        ts = feats[2] if len(feats) > 2 else _dtw.prepare(feats[1])
        costs = []
        for name, e in refs.items():
            ident = (name, len(e['words']), round(e['words'][0][0], 3), round(e['words'][-1][1], 3))
            if ident not in self._rstack:
                if len(self._rstack) > 64:
                    self._rstack.clear()
                rc, rf = api._collapse_ref(e)
                self._rstack[ident] = (rc, rf, _dtw.prepare(rf))
            rc, rf, full = self._rstack[ident]
            st, en = e['words'][j - 1][0], e['words'][k - 1][1]
            idx = np.flatnonzero((rc >= st - 0.02) & (rc <= en + 0.02))
            if len(idx) < 5:
                continue
            rs = _dtw.prepare_slice(rf, full, int(idx[0]), int(idx[-1]) + 1)
            costs.append(float(_dtw.dtw_path(_dtw.cost_matrix(rs, ts, prepared=True))[1]))
        return costs

    def _range_cost(self, api, feats, refs, j, k):
        """Mean DTW cost of the claim 'this audio holds words j..k' (1-based).
        Words that are not there are punished, and so are words left out --
        so the cost has a real minimum instead of always sliding to an answer."""
        costs = self._range_costs(api, feats, refs, j, k)
        return float(np.mean(costs)) if costs else 1e9

    def _word_share_ratio(self, api, med, refs):
        """How the aligned words divide the span, against how the reference
        reciters divide it. DTW must fill whatever window it is given, so a
        window cut short squeezes the closing words -- and every reference
        squeezes with it, which is why agreement cannot see truncation."""
        prof = []
        for entry in refs.values():
            ws = entry['words']
            tot = ws[-1][1] - ws[0][0]
            if tot > 0 and len(ws) == len(med):
                prof.append([(b - a) / tot for a, b in ws])
        if not prof:
            return None
        ref = np.mean(np.array(prof), axis=0)
        tot = float(med[-1][1] - med[0][0])
        if tot <= 0:
            return None
        got = np.array([float(b - a) / tot for a, b in med])
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(ref > 0, got / ref, 1.0)

    def _pause_candidates(self, wave, total, hop=0.025, min_run=0.15):
        """Breaths, measured against the recording's OWN noise floor: room tone
        sits near a tenth of peak, so a fixed fraction of the maximum finds
        nothing. Returns [(center, width)] with the speech start/end."""
        quiet, hop = self._quiet_mask(wave, total, hop)
        runs, i = [], 0
        while i < len(quiet):
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                runs.append((i * hop, j * hop))
                i = j
            else:
                i += 1
        voiced = np.where(~quiet)[0]
        if len(voiced) == 0:
            return [], 0.0, total
        speech_s, speech_e = voiced[0] * hop, (voiced[-1] + 1) * hop
        merged = []
        for lo, hi in runs:
            if merged and lo - merged[-1][1] < 0.15:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append([lo, hi])
        cands = [((lo + hi) / 2, hi - lo) for lo, hi in merged
                 if hi - lo >= min_run and lo > speech_s + 0.2 and hi < speech_e - 0.2]
        return cands, speech_s, speech_e

    def _sequential_bounds(self, wave, total, ayahs, prep, no_pause_penalty=0.55, dur_weight=0.3):
        """Boundaries for a contiguous run that fills the recording. The ayahs
        are in order, so segment the audio instead of hunting each one: prefer
        real breaths, weighted by how breath-like they are, and fall back to
        expected-duration interpolation where none fits -- a waqf is not an
        ayah end, so a split that makes the durations implausible loses."""
        cands, speech_s, speech_e = self._pause_candidates(wave, total)
        exp = np.array([max(prep[a]['exp_dur_raw'], 0.2) for a in ayahs], dtype=float)
        if exp.sum() <= 0 or speech_e - speech_s < 1.0:
            return None
        exp = exp / exp.sum() * (speech_e - speech_s)
        n = len(ayahs)
        # option list per boundary: every pause, plus "no pause here"
        # a wide breath is strong evidence; char-count durations are only a
        # weak prior, since elongation varies far more than text length
        opts = [(c, no_pause_penalty * (1.0 - min(w, 0.8) / 0.8)) for c, w in cands]

        # memoised over (ayah index, boundary position) and fanned out only to
        # pauses near the expected end -- plain recursion over every candidate
        # is exponential and hangs on a long recording
        memo = {}

        def solve(k, t0):
            """(cost, [boundaries]) for ayahs[k:] starting at t0"""
            key = (k, round(t0, 3))
            if key in memo:
                return memo[key]
            if k == n - 1:
                out = (dur_weight * abs(np.log(max(speech_e - t0, 0.05) / exp[k])), [])
                memo[key] = out
                return out
            rest = exp[k:].sum()
            interp = t0 + (speech_e - t0) * exp[k] / rest
            near = sorted((c for c, _ in opts if t0 + 0.3 < c < speech_e),
                          key=lambda c: abs(c - (t0 + exp[k])))[:MAX_SEQ_FANOUT]
            choices = [(c, pen) for c, pen in opts if c in near]
            choices.append((interp, no_pause_penalty))
            best = None
            for t1, pen in choices:
                if t1 <= t0 or t1 >= speech_e:
                    continue
                sub_cost, sub_path = solve(k + 1, t1)
                c = dur_weight * abs(np.log(max(t1 - t0, 0.05) / exp[k])) + pen + sub_cost
                if best is None or c < best[0]:
                    best = (c, [t1] + sub_path)
            out = best if best else (1e9, [])
            memo[key] = out
            return out

        cost, path = solve(0, speech_s)
        if not path and n > 1:
            return None
        bounds = [speech_s] + path + [speech_e]
        backed = [any(abs(t - c) < 0.15 for c, _ in cands) for t in path]
        return bounds, backed

    def _quiet_mask(self, wave, total, hop=0.025):
        """The one definition of silence, measured against the recording's OWN
        noise floor: room tone and reverb sit near a tenth of peak, so a fixed
        fraction of the maximum sits BELOW the noise and finds nothing."""
        env = wave[0][:int(total / hop) * int(hop * SR)].reshape(-1, int(hop * SR)).pow(2).mean(1).sqrt().numpy()
        floor = float(np.percentile(env, 10))
        return env < min(max(2.0 * floor, 0.02 * float(env.max())), 0.25 * float(env.max())), hop

    def _snap_to_silence(self, wave, total, s, e, max_ext=1.5, run=4):
        quiet, hop = self._quiet_mask(wave, total)

        def nudge(t, direction):
            # sustained quiet only: one dipping frame inside a fading tail is
            # not silence, and the recording's edges are silence by definition
            i = int(t / hop)
            for step in range(int(max_ext / hop)):
                j = i + direction * step
                if j < 0 or j >= len(quiet):
                    return min(max(0.0, j * hop), total)
                seg = quiet[j:j + run] if direction > 0 else quiet[max(0, j - run + 1):j + 1]
                if len(seg) and seg.all():
                    return j * hop
            return t
        return (max(0.0, nudge(s, -1)), min(total, nudge(e, +1)))

    def _nearest_pause(self, wave, total, t, radius=1.2, min_run=0.2):
        """Centre of the widest sustained pause within radius of t, else None.
        A boundary the reciter never stopped at is not a boundary."""
        quiet, hop = self._quiet_mask(wave, total)
        lo, hi = max(0, int((t - radius) / hop)), min(len(quiet), int((t + radius) / hop))
        best, i = None, lo
        while i < hi:
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                if (j - i) * hop >= min_run and (best is None or (j - i) > (best[1] - best[0])):
                    best = (i, j)
                i = j
            else:
                i += 1
        return None if best is None else (best[0] + best[1]) / 2 * hop

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

    def align(self, audio_path, surah, ayahs=None, ayah_start=None, ayah_end=None, canon_text=None, pace_hints=None, pairs_jsonl='auto', quran_api=DEFAULT_QURAN_API, residual_max_span=60.0, residual_max_run=10, mask_fatiha=True, decoy_margin=0.35, refine=True, sequential='auto', stops_path=True):
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
        # MAIN PATH: the reciter's own stops. The search below is the
        # fallback for when the segmenter is unavailable or cannot complete
        # the run (a recording that is not one continuous recitation).
        if stops_path:
            api_s = None
            try:
                from tadabur_align import WordTimestamps
                if self._refiner is None:
                    self._refiner = WordTimestamps(device=self.device, min_refs=2)
                api_s = self._refiner
            except Exception as ex:
                self._log(f'  aligner unavailable for the stop path ({ex})')
            if api_s is not None:
                try:
                    q0 = prep[ayahs[0]]['query']
                    sims0 = (chunk_embs @ q0).numpy()

                    def _span_score(i, _st=None):
                        a, b = self._stops_cache.get(audio_path, [(0, 0)])[i]
                        m = (chunk_centers >= a) & (chunk_centers <= b)
                        return float(sims0[m].max()) if m.any() else -1e9
                    out = self._align_by_stops(audio_path, wave, total, surah, ayahs, prep, api_s,
                                               span_score=_span_score)
                except Exception as ex:
                    self._log(f'  stop path failed ({ex}) -- falling back to the search')
                    out = None
                if out is not None:
                    self._log(f'total {time.time() - t0:.1f}s')
                    return out

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
        observed_pace = 1.0

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
        observed_pace = float(np.median(pace_ratios)) if pace_ratios else 1.0
        if skipped:
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
        # When the embedding cannot read this recitation (melismatic taraweeh
        # scores near noise), hunting each ayah separately scatters them. A
        # contiguous request is a run in order, so segment the audio instead.
        contiguous = ayahs == list(range(min(ayahs), max(ayahs) + 1))
        strong_hits = [a for a in ayahs if raw_results.get(a) and raw_results[a]['confident']
                       and raw_results[a]['sc'] >= self.quality_floor]
        use_seq = sequential is True or (sequential == 'auto' and contiguous and len(strong_hits) < len(ayahs) / 2)
        if use_seq and contiguous:
            seq = self._sequential_bounds(wave, total, ayahs, prep)
            if seq:
                bounds, backed = list(seq[0]), list(seq[1])
                self._log(f'  embedding evidence weak ({len(strong_hits)}/{len(ayahs)} above quality floor) -- '
                          f'segmenting the run by breaths instead')
                # a breath is not always an ayah end (the reciter may stop at a
                # waqf): a clip running far past its OWN words has swallowed the
                # next ayah, so trim it back to where its words actually end
                if refine:
                    try:
                        from tadabur_align import WordTimestamps
                        from tadabur_align import verify as ta_verify
                        if self._refiner is None:
                            self._refiner = WordTimestamps(device=self.device, min_refs=2)
                        api_s = self._refiner
                        for _round in range(2):
                            grids = {}
                            for i, ayah in enumerate(ayahs):
                                lo, hi = bounds[i], bounds[i + 1]
                                if hi - lo < 0.6:
                                    continue
                                try:
                                    med, mad, _, _, _ = ta_verify.word_stamps(api_s, wave[:, int(lo * SR):int(hi * SR)], surah, ayah)
                                    grids[i] = (lo + float(med[0][0]), lo + float(med[-1][1]))
                                except Exception:
                                    pass
                            if len(grids) < 3:
                                break
                            slack = {i: bounds[i + 1] - we for i, (ws, we) in grids.items()}
                            typical = float(np.median(list(slack.values())))
                            moved = False
                            for i in sorted(grids):
                                if i >= len(ayahs) - 1:
                                    continue
                                we = grids[i][1]
                                if slack[i] > max(1.0, 3 * max(typical, 0.05)) and we > bounds[i] + 0.5:
                                    nb = min(we + max(typical, 0.1), bounds[i + 1])
                                    self._log(f'  ayah {ayahs[i]}: clip runs {slack[i]:.2f}s past its own words '
                                              f'(typical {typical:.2f}s) -- boundary {bounds[i + 1]:.2f} -> {nb:.2f}')
                                    bounds[i + 1] = nb
                                    backed[i] = False
                                    moved = True
                            if not moved:
                                break
                    except Exception as ex:
                        self._log(f'  clip-trim pass unavailable ({ex})')
                entries = []
                for i, ayah in enumerate(ayahs):
                    st, en = float(bounds[i]), float(bounds[i + 1])
                    warns = []
                    if i > 0 and not backed[i - 1]:
                        warns.append('start_not_breath_backed_verify_by_ear')
                    if i < len(ayahs) - 1 and not backed[i]:
                        warns.append('end_not_breath_backed_verify_by_ear')
                    self._log(f'  ayah {ayah}: [{st:.2f},{en:.2f}] ({en - st:.2f}s) sequential'
                              f"{' -- ' + ','.join(warns) if warns else ''}")
                    entries.append(AyahResult(ayah, prep[ayah]['text'], round(st, 3), round(en, 3),
                                              True, 'sequential_breaths', 0.0, 0.0, 0.0, warns))
                self._log(f'done: {len(entries)}/{len(ayahs)} by sequential layout  (total {time.time() - t0:.1f}s)')
                return CropResult(entries, wave, SR, surah, audio_path)

        # consecutive ayahs of a continuous recitation join up: speech left
        # orphaned between two of them belongs to the later one, which simply
        # started too late. Only real speech counts -- a pause is not an orphan
        for ayah in ayahs:
            r, nx = raw_results.get(ayah), raw_results.get(ayah + 1)
            if not (r and nx and r['confident'] and nx['confident']):
                continue
            gap = nx['start'] - r['end']
            if gap <= 1.0:
                continue
            try:
                quiet, hop = self._quiet_mask(wave, total)
                seg = quiet[int(r['end'] / hop):int(nx['start'] / hop)]
                voiced_s = float((~seg).sum()) * hop if len(seg) else 0.0
            except Exception:
                voiced_s = 0.0
            if voiced_s < 0.5:
                continue
            back = self._nearest_pause(wave, total, r['end'] + 0.05, radius=0.5) or r['end']
            self._log(f'  ayah {ayah + 1}: {voiced_s:.1f}s of recitation was orphaned after ayah {ayah} '
                      f'-- start {nx["start"]:.2f} -> {back:.2f}')
            nx['start'] = back
            nx['warnings'] = nx['warnings'] + ['start_extended_over_orphaned_audio_verify_by_ear']

        # Boundary refinement: word timestamps transferred from reference
        # recitations (tadabur-align) replace loudness-based edges.
        if refine:
            api = None
            try:
                from tadabur_align import WordTimestamps
                from tadabur_align import verify as ta_verify
                if self._refiner is None:
                    self._refiner = WordTimestamps(device=self.device, min_refs=2)
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
                        # truncation audit: a reciter may stop at a waqf, and the
                        # crop then ends mid-ayah while every reference agrees on
                        # the squeezed layout. Closing words far short of their
                        # usual share mean the ending was cut off
                        share = self._word_share_ratio(api, med, own_refs)
                        # Where does this ayah BEGIN? Same question at the head:
                        # start too early and the first word stretches over the
                        # previous ayah's tail, too late and it collapses to zero
                        pinned_by_prev = bool(prev_r and prev_r.get('_repaired_end') is not None
                                              and abs(prev_r['_repaired_end'] - r['start']) < 0.05)
                        if share is not None and not pinned_by_prev:
                            exp_d0 = prep[ayah]['exp_dur_raw'] * max(observed_pace, 0.5)
                            floor_t = prev_r['start'] + 0.3 if prev_r else 0.0
                            cands_h, _, _ = self._pause_candidates(wave, total)
                            # only a NEARBY stop: a first-word share cannot tell a
                            # correct start from one seconds late (DTW refits
                            # either way), so this may correct a spill-over from
                            # the previous ayah, never relocate the ayah
                            behind = [c for c, _ in cands_h
                                      if floor_t < c < r['end'] - 0.35 * exp_d0 and abs(c - r['start']) <= 1.5]
                            behind.sort(key=lambda c: abs(c - r['start']))
                            sh_scored = []
                            for pt in behind[:10]:
                                try:
                                    med0, mad0, _, _, _ = ta_verify.word_stamps(api, wave[:, int(pt * SR):int(r['end'] * SR)], surah, ayah)
                                    sh0 = self._word_share_ratio(api, med0, own_refs)
                                    if sh0 is None:
                                        continue
                                    sh_scored.append((abs(np.log(max(sh0[0], 1e-3))) + float(mad0.mean()) / 300.0, pt, float(sh0[0]), float(mad0.mean())))
                                except Exception:
                                    continue
                            if sh_scored:
                                sh_scored.sort()
                                _, pt0, s0_v, m0_v = sh_scored[0]
                                if abs(pt0 - r['start']) >= 0.2 and 0.6 <= s0_v <= 1.5 and m0_v <= 200:
                                    self._log(f'  ayah {ayah}: begins at the stop at {pt0:.2f} (first word {s0_v:.2f} of its usual share, '
                                              f'agreement {m0_v:.0f}ms) -- was {r["start"]:.2f}')
                                    r['start'] = pt0
                                    r['_edge'] = (pt0, r['_edge'][1])
                                    r['_repaired_start'] = pt0
                                    if prev_r and prev_r['end'] > pt0:
                                        prev_r['end'] = pt0
                                        prev_r['warnings'] = prev_r['warnings'] + ['end_moved_by_neighbour_verify_by_ear']

                        # Where does this ayah actually end? The reciter's own
                        # stops are the only candidate cuts, and at each one the
                        # question is answerable: too short and the closing words
                        # are squeezed, too long and the last word stretches into
                        # the next ayah, right and the profile matches the
                        # references with their best agreement.
                        if share is not None:
                            exp_d = prep[ayah]['exp_dur_raw'] * max(observed_pace, 0.5)
                            room = min(total, next_r['end'] if next_r else total)
                            cands_p, _, _ = self._pause_candidates(wave, total)
                            ahead = [c for c, _ in cands_p
                                     if r['start'] + 0.35 * exp_d < c < min(room - 0.2, r['start'] + 3.5 * exp_d)]
                            # nearest the EXPECTED ending, not the first few in
                            # time: a slow reciter's true end can sit many stops
                            # further on, and a chronological cap never sees it
                            ahead.sort(key=lambda c: abs(c - (r['start'] + exp_d)))
                            scored = []
                            for pt in ahead[:14]:
                                try:
                                    med2, mad2, _, _, _ = ta_verify.word_stamps(api, wave[:, int(r['start'] * SR):int((pt + 0.25) * SR)], surah, ayah)
                                    sh2 = self._word_share_ratio(api, med2, own_refs)
                                    if sh2 is None:
                                        continue
                                    # agreement carries real weight: a garbled
                                    # alignment can produce any share profile
                                    scored.append((abs(np.log(max(sh2[-1], 1e-3))) + float(mad2.mean()) / 300.0, pt, float(sh2[-1]), float(mad2.mean())))
                                except Exception:
                                    continue
                            if scored:
                                scored.sort()
                                cost_v, pt, sh_v, mad_v = scored[0]
                                if abs(pt - r['end']) >= 0.2 and 0.6 <= sh_v <= 1.5 and mad_v <= 200:
                                    self._log(f'  ayah {ayah}: ends at the stop at {pt:.2f} (closing word {sh_v:.2f} of its usual share, '
                                              f'agreement {mad_v:.0f}ms) -- was {r["end"]:.2f}')
                                    r['end'] = pt
                                    r['_edge'] = (r['_edge'][0], pt)
                                    r['_repaired_end'] = pt
                                    if next_r and next_r['start'] < pt:
                                        next_r['start'] = pt
                                        next_r['warnings'] = next_r['warnings'] + ['start_moved_by_neighbour_verify_by_ear']
                                        self._log(f'    ayah {ayah + 1}: start pushed to {pt:.2f} (it held the previous ending)')
                                elif not (0.6 <= sh_v <= 1.5):
                                    r['warnings'] = r['warnings'] + ['ending_uncertain_verify_by_ear']
                                    self._log(f'  ayah {ayah}: no stop completes this ayah cleanly (best {sh_v:.2f} share at {pt:.2f}) -- verify by ear')
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
                        # the chain's OUTER edges are left alone on purpose: a
                        # four-ayah fit pins its middle and lets its two ends
                        # slide, and the single-ayah pass measured them tighter

                        for k, sp in enumerate(splits):
                            aa, bb = chain[k], chain[k + 1]
                            ra, rb = raw_results[aa], raw_results[bb]
                            # a boundary restored from a waqf truncation was
                            # proved against a pause and the word profile; this
                            # pass knows neither, so it may not move it back
                            if ra.get('_repaired_end') is not None or rb.get('_repaired_start') is not None:
                                self._log(f'  boundary {aa}|{bb}: left at {ra["end"]:.2f} (restored ending, not re-litigated)')
                                continue
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