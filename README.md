# tadabur-cropper

Crop individual ayahs (verses) out of a raw Quran recitation audio file, using audio-text embedding alignment instead of ASR or manual timestamping.

Given an audio file and a surah number, it returns per-ayah start/end times plus a confidence flag — no word-level transcription required, just an audio-text embedding model (see [tadabur-embedding](https://huggingface.co/FaisaI)) and a canonical verse text lookup.

## Install

```bash
pip install torch torchaudio soundfile numpy certifi transformers huggingface_hub recitations-segmenter
```

`matplotlib` is optional, only needed if you want the debug figure saved per ayah (`save(out_dir, with_figures=True)`, the default).

The embedding model and its alignment checkpoint both auto-download from [FaisaI/tadabur-embedding](https://huggingface.co/FaisaI/tadabur-embedding) on first use and cache locally — no manual setup needed. Pass `audio_model_path`/`ckpt_path` to `AyahAligner()` if you want to use your own instead.

## Quick start

```python
from ayah_aligner import AyahAligner

aligner = AyahAligner()                          # loads models once
result = aligner.align("recitation.mp3", surah=19, ayah_start=1, ayah_end=18)
result.save("out/")                              # writes manifest.json + one .wav (+ figure) per confident ayah

for entry in result:
    print(entry.ayah, entry.confident, entry.start_s, entry.end_s)
```

Or one-shot:

```python
from ayah_aligner import crop_ayahs
result = crop_ayahs("recitation.mp3", surah=19, ayah_start=1, ayah_end=18, out_dir="out/")
```

Ayah selection: pass `ayahs=[...]` for an explicit/non-contiguous list, `ayah_start`/`ayah_end` for a contiguous range, or neither for the whole surah.

## How it works, briefly

- Ayahs are resolved in number order. Each one tries a staged sequence of anchors — first the immediately-preceding confirmed ayah's end, then a global audio-text similarity search as a stricter fallback.
- Candidate boundaries are built from real pauses the recitation-segmenter detects (`obadx/recitation-segmenter-v2`, auto-downloaded), merged into runs and scored by whole-clip audio-text similarity — not word-level ASR.
- Every result carries a `confident` flag and `warnings`. **Ear-checking the output is the real ground truth this pipeline serves** — it narrows a full recording down to a short, mostly-correct list of candidates to verify, not a replacement for checking them.
- Per-ayah pacing (`pace_hints`) is auto-downloaded and cached from a companion dataset on first use, so duration estimates are calibrated per-ayah rather than one flat rate for everything.

The full design rationale (why each specific check exists, what real failure case it was built to catch) lives in the docstrings in `boundary_refine.py` and `ayah_aligner.py` — read those before changing scoring logic.

## 📄 Licenses

| Component                                                            | License                                  |
| -------------------------------------------------------------------- | ---------------------------------------- |
| This code                                                            | MIT                                      |
| [tadabur-embedding](https://huggingface.co/FaisaI/tadabur-embedding) | CC BY-NC 4.0                             |
| [quran-align](https://github.com/cpfair/quran-align) timestamps      | CC BY 4.0                                |
| Reference audio ([everyayah.com](https://everyayah.com))             | fetched at build time, not redistributed |

Part of the tadabur family: [dataset](https://huggingface.co/datasets/FaisaI/tadabur) · [embedding](https://huggingface.co/FaisaI/tadabur-embedding) · [align](https://github.com/FaisaI/tadabur-align) · this cropper.
