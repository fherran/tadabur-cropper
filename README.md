# tadabur-cropper

Crop individual ayahs (verses) out of a raw Quran recitation audio file, using audio-text embedding alignment instead of ASR or manual timestamping.

Given an audio file and a surah number, it returns per-ayah start/end times plus a confidence flag — no word-level transcription required, just an audio-text embedding model (see [tadabur-embedding](https://huggingface.co/FaisaI)) and a canonical verse text lookup.

## Install

```bash
pip install git+https://github.com/fherran/tadabur-cropper
```

This installs [tadabur-align](https://github.com/FaisaI/tadabur-align) with it — required for boundary refinement and verification (the `refine` stage). All models and data auto-download on first use and cache locally. `matplotlib` is optional (debug figures).

## Quick start

```python
from tadabur_cropper import crop_ayahs

result = crop_ayahs("recitation.mp3", surah=19, ayah_start=1, ayah_end=18, out_dir="out/")
```

Or keep the models loaded across many files:

```python
from tadabur_cropper import AyahCropper

cropper = AyahCropper(device="cuda")             # loads models once; or device="cpu" (None auto-detects)
result = cropper.align("recitation.mp3", surah=19, ayah_start=1, ayah_end=18)
result.save("out/")
```

Ayah selection: `ayahs=[...]` for an explicit list, `ayah_start`/`ayah_end` for a range, or neither for the whole surah.

Surah numbers: `surah` is the surah's number (1-114). If you are not sure of a number, read [`surah_index.json`](surah_index.json) in the repo root: one entry per surah with `surah_id`, the English and Arabic names, and `totalAyah`.

## What you get back

One entry per ayah, in `result` and in `out/manifest.json`, plus one `.wav` per confident ayah (cut from the original file at full quality):

```python
for e in result:
    print(e.ayah, e.confident, e.start_s, e.end_s, e.refine_mad_ms, e.warnings)
```

How to read it:

- **`confident=True`, no warnings** — trust it.
- **`confident=True` + a `verify_by_ear` warning** — probably right; listen once before relying on it.
- **`confident=False`** — the tool could not place this ayah honestly. The timestamps (if any) are a best guess, and the warnings say why: e.g. `differing_word_matches_rival_26:161` means the audio at that spot is actually a different, near-identical ayah — named for you.
- **`refine_mad_ms`** — how closely ~7 reference reciters agreed on the word timings (small = tight boundaries).

**Ear-checking flagged results is the real ground truth this pipeline serves** — it narrows a full recording down to a short list of candidates to verify, not a replacement for checking them.

## How it works

1. **Search** — each ayah is placed by audio-text similarity: anchored right after its confirmed neighbor when possible, otherwise by a stricter whole-file search. Candidate boundaries come from real pauses (`obadx/recitation-segmenter-v2`), and expected durations are calibrated per-ayah from a companion dataset plus the reciter's own observed pace.
2. **Order check** — the Quran's ayah order is fixed; any placement that contradicts confirmed neighbors is demoted, never silently kept.
3. **Al-Fatiha guard** — prayer recordings recite al-Fatiha mid-file; a span matching it decisively better than the requested ayah is rejected.
4. **Boundary refinement** (tadabur-align) — word start/end times transferred from ~7 reference reciters replace loudness-based edges. Recovers softly-spoken opening words; splits ayah pairs recited with no pause between them.
5. **Mutashabihat check** (tadabur-align) — `mutashabihat.json` lists every ayah's near-identical rivals Quran-wide (1,887 pairs, built once from the text by `mutashabihat/build_mutashabihat.py`). For those ayahs, the differing word's audio is compared against both candidates' references: a decisive rival win rejects the crop and names the true ayah; a close call is flagged; it never guesses.

## Arguments reference

### `AyahCropper(...)` — one-time setup

| Argument | What it means |
|---|---|
| `audio_model_path`, `ckpt_path` | Embedding model + checkpoint. Default: auto-download from Hugging Face. |
| `text_model_name` | Text embedding model for the verse text. |
| `quality_floor` | Minimum match quality to call a result confident. |
| `fallback_floor` | Stricter floor for whole-file searches (no neighbor anchor). |
| `default_rate` | Seconds-per-character fallback when no pacing data exists for an ayah. |
| `min_dur` | Shortest allowed crop, in seconds. |
| `verbose` | Print progress. |
| `device`, `segmenter_device` | `"cpu"` / `"cuda"` / `"mps"`; `None` auto-detects. |

### `align(...)` / `crop_ayahs(...)` — per run

| Argument | What it means |
|---|---|
| `audio_path`, `surah` | The file and its surah number (1-114; see `surah_index.json` for names). |
| `ayahs` or `ayah_start`/`ayah_end` | Which ayahs; omit for the whole surah. |
| `canon_text` | Verse text if you have it; `None` fetches it. |
| `pace_hints`, `pairs_jsonl` | Per-ayah pacing data; defaults auto-download it. |
| `quran_api` | Where verse text is fetched from. |
| `residual_max_span`, `residual_max_run` | Backstops on last-resort guesses (60s / 10 segments); the real bound is the reciter's observed pace. |
| `mask_fatiha`, `decoy_margin` | The al-Fatiha guard and how decisively it must win (0.35). |
| `refine` | Boundary refinement + mutashabihat check. On by default; skipped gracefully if tadabur-align isn't installed. |
| `out_dir`, `aligner` (`crop_ayahs` only) | Output folder; reuse a loaded `AyahCropper`. |

## 📄 Licenses

| Component                                                                                          | License                                  |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| This code                                                                                          | MIT                                      |
| [tadabur-embedding](https://huggingface.co/FaisaI/tadabur-embedding)                               | CC BY-NC 4.0                             |
| [tadabur-align-references](https://huggingface.co/datasets/FaisaI/tadabur-align-references) (reference features and word timestamps, fetched at runtime) | CC BY-NC 4.0 |
| Reference audio ([everyayah.com](https://everyayah.com))                                           | fetched at build time, not redistributed |
| Recitation segmenter ([obadx/recitation-segmenter-v2](https://huggingface.co/obadx/recitation-segmenter-v2)) | see model card                   |

Part of the tadabur family: [dataset](https://huggingface.co/datasets/FaisaI/tadabur) · [embedding](https://huggingface.co/FaisaI/tadabur-embedding) · [align](https://github.com/fherran/tadabur-align) · this cropper.
