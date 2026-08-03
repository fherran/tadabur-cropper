"""Boundary verification for the ayah cropper: DTW polish against dataset
reference recordings + an independent recitation-segmenter cross-check.

Usage from a notebook:

    import boundary_refine as br

    s, e = crop_ayah(text, expected_dur)          # from crop_alignment.ipynb
    result = br.refine_boundaries(audio_model, wave, TOTAL, s, e, SURAH, ayah)
    s, e = result["start"], result["end"]
    if result["warnings"]:
        print(ayah, result["warnings"])

Both checks are independent verifiers, not part of the core cropper (see
ayah_aligner.py for that) -- crop_ayahs()/AyahAligner.align() do not call
into this module's DTW/reference-lookup functions at all:
- DTW compares the crop's frame-level content against 2-3 real recordings
  of the same ayah (by distinct reciters), then tightens the boundaries.
  Requires PAIRS and AUDIO_DIR (below) to be set to your own dataset --
  there is no bundled reference-audio dataset, unlike the pace-hints
  dataset ayah_aligner.py auto-downloads.
- The segmenter (obadx/recitation-segmenter-v2 -- the same model used to
  build the tadabur dataset) hears natural pauses, including short
  mid-ayah breaths, and flags if it disagrees with the DTW result. This
  one needs no extra setup (the segmenter model auto-downloads).
Convergence between the two is what makes a boundary trustworthy -- a
single method (even averaged over several references) can share a blind
spot.
"""
import json
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
SR = 16000
PAIRS = None
AUDIO_DIR = None
_FILENAME_RE = re.compile('tadabur_spk(\\d+)_S(\\d+)_A(\\d+)_([0-9a-f]+)_\\d+\\.wav')
_by_ayah = None

def _load_pairs_index():
    global _by_ayah
    if _by_ayah is not None:
        return _by_ayah
    _by_ayah = {}
    if not PAIRS:
        return _by_ayah
    with open(PAIRS, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            _by_ayah.setdefault((d['surah_id'], d['ayah_id']), []).append(d)
    return _by_ayah

def find_references(surah_true, ayah, n=3):
    rows = _load_pairs_index().get((surah_true - 1, ayah), [])
    seen, refs = (set(), [])
    for d in sorted(rows, key=lambda d: -len(d['text'].replace(' ', ''))):
        m = _FILENAME_RE.match(d['audio_filename'])
        spk = m.group(1) if m else d['audio_filename']
        if spk in seen:
            continue
        seen.add(spk)
        refs.append(d)
        if len(refs) >= n:
            break
    return refs

def _frame_embed(audio_model, w):
    w = w - w.mean()
    m = torchaudio.compliance.kaldi.fbank(w, htk_compat=True, sample_frequency=SR, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    nf = m.shape[0]
    m = F.pad(m, (0, 0, 0, 1024 - nf)) if nf < 1024 else m[:1024]
    m = (m - -4.381) / (3.628 * 2)
    with torch.no_grad():
        feats = audio_model.extract_features(m[None, None])
    return (F.normalize(feats[0, 1:].reshape(64, 8, 768).mean(1), dim=-1), min(64, int(np.ceil(min(nf, 1024) / 16))))

def _subseq_dtw(Fr, Fv):
    S = (Fr @ Fv.T).numpy()
    n, m = S.shape
    cost = 1.0 - S
    D = np.full((n + 1, m + 1), np.inf)
    D[0, :] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    j_end = int(np.argmin(D[n, 1:])) + 1
    i, j, path = (n, j_end, [])
    while i > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if k == 0:
            i, j = (i - 1, j - 1)
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return (float(np.mean([S[a, b] for a, b in path])), path[0][1], path[-1][1])

def dtw_polish(audio_model, wave, total, s, e, references):
    if not references:
        return (s, e, None, None, None)
    mid = (s + e) / 2
    w_end = min(total, mid + 5.12)
    w_start = max(0.0, w_end - 10.24)
    Fv, nv = _frame_embed(audio_model, wave[:, int(w_start * SR):int(w_end * SR)])
    Fv = Fv[:nv]
    results = []
    for d in references:
        wav_r, sr_r = sf.read(os.path.join(AUDIO_DIR, d['audio_filename']), dtype='float32')
        wr = torch.from_numpy(wav_r.T if wav_r.ndim > 1 else wav_r[None]).mean(0, keepdim=True)
        a, b = (int(d['start_ms'] * sr_r / 1000), int(d['end_ms'] * sr_r / 1000))
        wr = wr[:, a:min(b, wr.shape[1])]
        if sr_r != SR:
            wr = torchaudio.functional.resample(wr, sr_r, SR)
        Fr, nr = _frame_embed(audio_model, wr)
        sim, j0, j1 = _subseq_dtw(Fr[:nr], Fv)
        t0, t1 = (w_start + j0 * 0.16, w_start + (j1 + 1) * 0.16)
        edge = j0 == 0 or j1 >= nv - 1
        tail = float((Fr[max(0, nr - 6):nr] @ Fv[max(0, j1 - 5):j1 + 1].T).diagonal().mean())
        results.append((sim, t0, t1, edge, tail))
    ok = [r for r in results if not r[3]]
    sim, t0, t1, edge, tail = max(ok if ok else results)
    return (max(0.0, t0 - 0.15), min(total, t1 + 0.15), sim, tail, len(ok) == 0)
_segmenter = None
_DEVICE = 'cpu'

def set_device(device):
    global _DEVICE, _segmenter
    _DEVICE = device
    if _segmenter is not None:
        fe, m = _segmenter
        _segmenter = (fe, m.to(device))

def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
        fe = AutoFeatureExtractor.from_pretrained('obadx/recitation-segmenter-v2')
        m = AutoModelForAudioFrameClassification.from_pretrained('obadx/recitation-segmenter-v2').to(_DEVICE, dtype=torch.float32).eval()
        _segmenter = (fe, m)
    return _segmenter

def segmenter_check(wave, total, s, e, pad_before=8.0, pad_after=20.0):
    from recitations_segmenter import clean_speech_intervals, segment_recitations
    from recitations_segmenter.segment import NoSpeechIntervals
    fe, seg_model = _get_segmenter()
    lo, hi = (max(0.0, s - pad_before), min(total, e + pad_after))
    region = wave[0, int(lo * SR):int(hi * SR)]
    out = segment_recitations([region], seg_model, fe, device=_DEVICE, dtype=torch.float32, batch_size=8)
    try:
        co = clean_speech_intervals(out[0].speech_intervals, out[0].is_complete, min_silence_duration_ms=30, min_speech_duration_ms=30, pad_duration_ms=30, return_seconds=True)
    except NoSpeechIntervals:
        return None
    segs = [(lo + a, lo + b) for a, b in (iv.tolist() for iv in co.clean_speech_intervals)]
    covering = [seg for seg in segs if seg[1] > s and seg[0] < e]
    if not covering:
        return None
    seg_start, seg_end = (min((x[0] for x in covering)), max((x[1] for x in covering)))
    later = [x[0] for x in segs if x[0] >= seg_end]
    next_start = min(later) if later else hi
    seg_end = extend_to_natural_decay(wave, seg_end, next_start)
    return (seg_start, seg_end)

def extend_to_natural_decay(wave, e, next_boundary, hop=0.025, floor_window=0.6, confirm_s=0.12, margin=1.6):
    n_hop = int(hop * SR)
    lo_i, hi_i = (int(e * SR), int(next_boundary * SR))
    region = wave[0, lo_i:hi_i]
    n = region.shape[0] // n_hop * n_hop
    if n < n_hop:
        return e
    env = region[:n].reshape(-1, n_hop).pow(2).mean(1).sqrt().numpy()
    floor_frames = max(1, int(floor_window / hop))
    noise_floor = float(np.median(env[-floor_frames:])) if len(env) > floor_frames else float(env.min())
    thresh = noise_floor * margin
    confirm_n = max(1, int(confirm_s / hop))
    quiet_run = 0
    stop_i = len(env)
    for i, v in enumerate(env):
        if v <= thresh:
            quiet_run += 1
            if quiet_run >= confirm_n:
                stop_i = i - confirm_n + 1
                break
        else:
            quiet_run = 0
    return min(next_boundary, e + stop_i * hop)

def edge_match_scores(embed_text_fn, embed_clips_fn, text, s, e, edge_dur=2.5):
    words = text.split()
    n = max(2, int(len(words) * 0.4))
    q_open = embed_text_fn(' '.join(words[:n]))
    q_close = embed_text_fn(' '.join(words[-n:]))
    open_span = (s, min(e, s + edge_dur))
    close_span = (max(s, e - edge_dur), e)
    embs = embed_clips_fn([open_span, close_span])
    return (float(embs[0] @ q_open), float(embs[1] @ q_close))

def windowed_similarity(embed_clips_fn, s, e, query, win=10.0, stride=5.0):
    if e - s <= 10.2:
        return float((embed_clips_fn([(s, e)]) @ query)[0])
    starts = list(np.arange(s, e - win + 1e-06, stride)) or [s]
    if starts[-1] + win < e:
        starts.append(e - win)
    spans = [(a, min(e, a + win)) for a in starts]
    sims = (embed_clips_fn(spans) @ query).numpy()
    return float(sims.max())

def segment_run_candidates(embed_text_fn, embed_clips_fn, text, query, segs, anchor, exp_dur, max_run=4, edge_dur=2.5, other_queries=None, decisive_margin=0.2):
    words = text.split()
    n = max(2, int(len(words) * 0.4))
    q_open = embed_text_fn(' '.join(words[:n]))
    q_close = embed_text_fn(' '.join(words[-n:]))
    near = [i for i, seg in enumerate(segs) if seg[0] - exp_dur - 2.0 <= anchor <= seg[1] + exp_dur + 2.0]
    out = []
    for i in near:
        s = segs[i][0]
        open_span = (s, min(segs[i][1], s + edge_dur))
        open_score = float((embed_clips_fn([open_span]) @ q_open)[0])
        anchor_bonus = 0.05 if s <= anchor <= segs[i][1] else 0.0
        for j in range(i, min(i + max_run, len(segs))):
            if j > i and other_queries:
                seg_span = (segs[j][0], segs[j][1])
                seg_sc_here = windowed_similarity(embed_clips_fn, *seg_span, query)
                seg_sc_elsewhere = max((windowed_similarity(embed_clips_fn, *seg_span, oq) for oq in other_queries))
                if seg_sc_elsewhere - seg_sc_here > decisive_margin:
                    break
            e_ext, e_raw = (segs[j][1], segs[j][2])
            close_span = (max(s, e_raw - edge_dur), e_raw)
            close_score = float((embed_clips_fn([close_span]) @ q_close)[0])
            sc = windowed_similarity(embed_clips_fn, s, e_ext, query)
            out.append({'i': i, 'j': j, 'start': s, 'end': e_ext, 'open_score': open_score, 'close_score': close_score, 'anchor_bonus': anchor_bonus, 'sc': sc})
    return out

def pick_best_candidate(scored, tolerance=0.08):
    best_score = max((x[3] for x in scored))
    close = [x for x in scored if x[3] >= best_score - tolerance]
    return max(close, key=lambda x: x[2] - x[1])

def get_segments(wave, total, lo, hi, pad=8.0):
    from recitations_segmenter import clean_speech_intervals, segment_recitations
    from recitations_segmenter.segment import NoSpeechIntervals
    fe, seg_model = _get_segmenter()
    lo2, hi2 = (max(0.0, lo - pad), min(total, hi + pad))
    region = wave[0, int(lo2 * SR):int(hi2 * SR)]
    out = segment_recitations([region], seg_model, fe, device=_DEVICE, dtype=torch.float32, batch_size=8)
    try:
        co = clean_speech_intervals(out[0].speech_intervals, out[0].is_complete, min_silence_duration_ms=30, min_speech_duration_ms=30, pad_duration_ms=30, return_seconds=True)
    except NoSpeechIntervals:
        return []
    segs = [(lo2 + a, lo2 + b) for a, b in (iv.tolist() for iv in co.clean_speech_intervals)]
    out_segs = []
    for i, (a, b) in enumerate(segs):
        next_start = segs[i + 1][0] if i + 1 < len(segs) else hi2
        out_segs.append((a, extend_to_natural_decay(wave, b, next_start), b))
    return out_segs

def finalize_start(wave, total, s, e, search=15.0):
    from recitations_segmenter import clean_speech_intervals, segment_recitations
    from recitations_segmenter.segment import NoSpeechIntervals
    fe, seg_model = _get_segmenter()
    lo, hi = (max(0.0, s - search), min(total, e + 3.0))
    region = wave[0, int(lo * SR):int(hi * SR)]
    out = segment_recitations([region], seg_model, fe, device=_DEVICE, dtype=torch.float32, batch_size=8)
    try:
        co = clean_speech_intervals(out[0].speech_intervals, out[0].is_complete, min_silence_duration_ms=30, min_speech_duration_ms=30, pad_duration_ms=30, return_seconds=True)
    except NoSpeechIntervals:
        return s
    segs = [(lo + a, lo + b) for a, b in (iv.tolist() for iv in co.clean_speech_intervals)]
    earlier = [x[1] for x in segs if x[1] <= s]
    prev_end = max(earlier) if earlier else lo
    return extend_start_to_natural_onset(wave, s, prev_end)

def extend_start_to_natural_onset(wave, s, prev_boundary, hop=0.025, floor_window=0.6, confirm_s=0.12, margin=1.6):
    n_hop = int(hop * SR)
    lo_i, hi_i = (int(prev_boundary * SR), int(s * SR))
    region = wave[0, lo_i:hi_i]
    n = region.shape[0] // n_hop * n_hop
    if n < n_hop:
        return s
    env = region[-n:].reshape(-1, n_hop).pow(2).mean(1).sqrt().numpy()[::-1]
    floor_frames = max(1, int(floor_window / hop))
    noise_floor = float(np.median(env[-floor_frames:])) if len(env) > floor_frames else float(env.min())
    thresh = noise_floor * margin
    confirm_n = max(1, int(confirm_s / hop))
    quiet_run = 0
    stop_i = len(env)
    for i, v in enumerate(env):
        if v <= thresh:
            quiet_run += 1
            if quiet_run >= confirm_n:
                stop_i = i - confirm_n + 1
                break
        else:
            quiet_run = 0
    return max(prev_boundary, s - stop_i * hop)

def finalize_end(wave, total, s, e, search=15.0):
    from recitations_segmenter import clean_speech_intervals, segment_recitations
    from recitations_segmenter.segment import NoSpeechIntervals
    fe, seg_model = _get_segmenter()
    lo, hi = (max(0.0, s - 3.0), min(total, e + search))
    region = wave[0, int(lo * SR):int(hi * SR)]
    out = segment_recitations([region], seg_model, fe, device=_DEVICE, dtype=torch.float32, batch_size=8)
    try:
        co = clean_speech_intervals(out[0].speech_intervals, out[0].is_complete, min_silence_duration_ms=30, min_speech_duration_ms=30, pad_duration_ms=30, return_seconds=True)
    except NoSpeechIntervals:
        return e
    segs = [(lo + a, lo + b) for a, b in (iv.tolist() for iv in co.clean_speech_intervals)]
    later = [x[0] for x in segs if x[0] >= e]
    next_start = min(later) if later else hi
    return extend_to_natural_decay(wave, e, next_start)

def overlaps_any(s, e, spans, pad=0.3, frac=0.25):
    for cs, ce in spans:
        ov = max(0.0, min(e, ce + pad) - max(s, cs - pad))
        if ov / max(1e-06, min(e - s, ce - cs)) > frac:
            return True
    return False

def find_gaps(confirmed_spans, total, min_gap=3.0):
    spans = sorted(confirmed_spans)
    gaps, prev_end = ([], 0.0)
    for s, e in spans:
        if s - prev_end > min_gap:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if total - prev_end > min_gap:
        gaps.append((prev_end, total))
    return gaps

def _coarse_window(embed_clips_fn, query, total, exp_dur, anchor, min_dur=0.4):
    lo = max(0.0, anchor - exp_dur)
    hi = min(total, anchor + 2 * exp_dur)
    starts = np.arange(lo, max(hi - min_dur, lo) + 1e-06, 0.25)
    cand = [(s, min(s + exp_dur, total)) for s in starts if min(s + exp_dur, total) - s >= min_dur]
    if not cand:
        return (anchor, min(total, anchor + exp_dur))
    scores = (embed_clips_fn(cand) @ query).numpy()
    s, e = cand[int(scores.argmax())]
    for _ in range(2):
        cs = [max(lo, s + d) for d in np.arange(-1.5, 1.51, 0.2) if lo <= max(lo, s + d) < e - min_dur]
        if cs:
            s = cs[int((embed_clips_fn([(x, e) for x in cs]) @ query).numpy().argmax())]
        ce = [min(hi, e + d) for d in np.arange(-1.5, 1.51, 0.2) if min(hi, e + d) > s + min_dur]
        if ce:
            e = ce[int((embed_clips_fn([(s, x) for x in ce]) @ query).numpy().argmax())]
    return (s, e)

def _candidates_from_anchors(text, query, wave, total, embed_text_fn, embed_clips_fn, anchors, exp_dur, exp_dur_raw, max_run, other_queries=None):
    out = []
    for anchor in anchors:
        radius = max(15.0, 2 * exp_dur)
        lo_r, hi_r = (max(0.0, anchor - radius), min(total, anchor + radius))
        segs = get_segments(wave, total, lo_r, hi_r)
        segrun = segment_run_candidates(embed_text_fn, embed_clips_fn, text, query, segs, anchor, exp_dur, max_run=max_run, other_queries=other_queries)
        for f in segrun:
            out.append((f"segrun[{f['i']}:{f['j']}]", f['start'], f['end'], f['sc'], f['open_score'], f['close_score'], f['anchor_bonus'], segs))
        if not segrun:
            cs, ce = _coarse_window(embed_clips_fn, query, total, exp_dur, anchor)
            co, cc = edge_match_scores(embed_text_fn, embed_clips_fn, text, cs, ce)
            coarse_sc = windowed_similarity(embed_clips_fn, cs, ce, query)
            out.append(('coarse', cs, ce, coarse_sc, co, cc, 0.0, segs))
    return out

def resolve_ayah(text, query, wave, total, embed_text_fn, embed_clips_fn, anchor_groups, exp_dur, exp_dur_raw, confirmed_spans, quality_floor=0.45, fallback_floor=0.75, max_run=4, other_queries=None):

    def rank(c):
        dur = c[2] - c[1]
        over = max(0.0, dur - 1.4 * exp_dur_raw)
        penalty = over / max(exp_dur_raw, 3.0) * 0.6
        return c[3] + 0.3 * c[5] + c[6] - penalty
    best_overall, best_floor, blocked = (None, quality_floor, True)
    for group_idx, anchors in enumerate(anchor_groups):
        floor = quality_floor if group_idx == 0 else max(quality_floor, fallback_floor)
        cands = _candidates_from_anchors(text, query, wave, total, embed_text_fn, embed_clips_fn, anchors, exp_dur, exp_dur_raw, max_run, other_queries=other_queries)
        clean = [c for c in cands if not overlaps_any(c[1], c[2], confirmed_spans)]
        if best_overall is None and cands:
            best_overall, best_floor = (max(cands, key=rank), floor)
        if not clean:
            continue
        blocked = False
        candidate = max(clean, key=rank)
        if rank(candidate) >= floor:
            best_overall, best_floor = (candidate, floor)
            break
        if best_overall is None or rank(candidate) > rank(best_overall):
            best_overall, best_floor = (candidate, floor)
    if best_overall is None:
        return {'method': 'none', 'start': anchor_groups[0][0] if anchor_groups and anchor_groups[0] else 0.0, 'end': 0.0, 'sc': 0.0, 'open_score': 0.0, 'close_score': 0.0, 'confident': False, 'warnings': ['no_candidates_generated']}
    name, s, e, sc, o, c, ab, segs = best_overall
    quality = sc + 0.3 * c + ab
    confident = quality >= best_floor and (not blocked)
    warnings = []
    if blocked:
        warnings.append('all_candidates_overlapped_confirmed_neighbor')
    elif not confident:
        warnings.append('below_quality_floor')
    return {'method': name, 'start': s, 'end': e, 'sc': sc, 'open_score': o, 'close_score': c, 'confident': confident, 'warnings': warnings}

def refine_boundaries(audio_model, wave, total, s, e, surah_true, ayah, use_segmenter=True):
    references = find_references(surah_true, ayah)
    t0, t1, sim, tail, hit_edge = dtw_polish(audio_model, wave, total, s, e, references)
    seg = segmenter_check(wave, total, t0, t1) if use_segmenter else None
    seg_disagrees = seg is not None and (abs(seg[0] - t0) > 0.3 or abs(seg[1] - t1) > 0.3)
    warnings = []
    if not references:
        warnings.append('no_dtw_reference_available')
    if hit_edge:
        warnings.append('dtw_window_edge_hit_low_confidence')
    if tail is not None and tail < 0.3:
        warnings.append('weak_tail_match_possible_truncation')
    if seg_disagrees:
        warnings.append('segmenter_disagrees_with_dtw_by_over_0.3s')
    return {'start': t0, 'end': t1, 'dtw_sim': round(sim, 4) if sim is not None else None, 'dtw_n_references': len(references), 'tail_match': round(tail, 4) if tail is not None else None, 'segmenter_boundaries': seg, 'warnings': warnings}