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

## Arguments, in plain words

### `AyahAligner(...)` — one-time setup

| Argument | What it means |
|---|---|
| `audio_model_path` | Where to load the audio embedding model from. Defaults to the public model on Hugging Face — downloads automatically. |
| `ckpt_path` | Where to load the alignment checkpoint from. Defaults to `"auto"`, which downloads it automatically. |
| `text_model_name` | Which text embedding model reads the verse text. Defaults to a general Arabic-capable model. |
| `quality_floor` | How confident a match has to be before the tool trusts its own best guess. Lower = more results marked confident, but more likely to include mistakes. |
| `fallback_floor` | A stricter version of `quality_floor`, used only when the tool couldn't anchor off the neighboring ayah and had to search the whole recording instead. |
| `default_rate` | Seconds-per-character to assume when there's no real pacing data for an ayah. Fallback only. |
| `min_dur` | The shortest a cropped ayah is allowed to be, in seconds — prevents absurdly tiny crops. |
| `verbose` | Print progress messages while running. Set `False` for silent operation. |
| `device` | Which hardware runs the embedding models (`"cpu"`, `"mps"`, `"cuda"`). Leave `None` to auto-detect the best available. |
| `segmenter_device` | Same idea, just for the pause-detection model. Leave `None` to match `device`. |

### `aligner.align(...)` — run it on one audio file

| Argument | What it means |
|---|---|
| `audio_path` | Path to the audio file to crop. |
| `surah` | Which surah (chapter) number this recitation is from. |
| `ayahs` | An explicit list of ayah numbers to crop (e.g. `[1, 3, 7]`), instead of a simple range. |
| `ayah_start` / `ayah_end` | Crop a continuous range of ayahs. Leave both out to crop the whole surah. |
| `canon_text` | The official verse text, if you already have it. Leave `None` to fetch it automatically. |
| `pace_hints` | Your own per-ayah pacing data, if you have it. Leave `None` to use the automatic data (recommended). |
| `pairs_jsonl` | Where the automatic pacing data comes from. `"auto"` downloads it; `None` skips pacing calibration entirely (less accurate). |
| `quran_api` | Where to fetch the official verse text from, if `canon_text` isn't given. |
| `residual_max_span` | A generous backstop, in seconds, on how long a "last resort" guess is allowed to be. The real bound is now adaptive (the reciter's own observed pace). Default 60s. |
| `residual_max_run` | Backstop on how many separate audio segments can be glued into one last-resort guess. Default 10. |
| `mask_fatiha` | Prayer recordings recite al-Fatiha mid-file; spans matching it decisively better than the requested ayah are rejected. Default on. |
| `decoy_margin` | How decisively al-Fatiha must win before a span is rejected. Default 0.35. |
| `refine` | Polish every confident crop's boundaries with word timestamps from reference recitations ([tadabur-align](https://github.com/FaisaI/tadabur-align)), and verify mutashabihat ayahs by dueling the differing word's audio against each rival's references. Default on; skipped gracefully if tadabur-align isn't installed. |

### `crop_ayahs(...)` — one-shot shortcut

Same `audio_path`/`surah`/`ayahs`/`ayah_start`/`ayah_end` as above, plus:

| Argument | What it means |
|---|---|
| `out_dir` | Folder to save the cropped `.wav` files (and manifest) into. |
| `aligner` | Reuse an already-created `AyahAligner` instead of loading the models again — faster when cropping many files. |
| (any other keyword) | Passed straight through to `AyahAligner(...)` when it creates a new one. |

## Verification layers

Beyond the embedding search, four independent checks guard every result:

- **Order consistency** -- the Quran's ayah order is fixed; a placement that contradicts confirmed neighbors is demoted, never silently kept.
- **Al-Fatiha decoys** (`mask_fatiha`) -- for prayer recordings.
- **Boundary refinement** (`refine`) -- word-level start/end times transferred from ~7 reference reciters replace loudness-based edges (recovers soft opening words; splits ayah pairs recited without a pause). Each entry reports `refine_mad_ms`, how closely the references agreed.
- **Mutashabihat duel** (`refine`) -- `mutashabihat.json` (built once from the Quran's text by `mutashabihat/build_mutashabihat.py`, 1,887 pairs Quran-wide) lists every ayah's near-identical rivals and where they differ. For those ayahs the differing word's audio is compared against both candidates' reference recordings: a decisive rival win rejects the crop naming the true ayah; a close call is flagged `verify_by_ear`; it never guesses.

Boundary refinement and the duel need `pip install git+https://github.com/FaisaI/tadabur-align`.

## How it works, briefly

- Ayahs are resolved in number order. Each one tries a staged sequence of anchors — first the immediately-preceding confirmed ayah's end, then a global audio-text similarity search as a stricter fallback.
- Candidate boundaries are built from real pauses the recitation-segmenter detects (`obadx/recitation-segmenter-v2`, auto-downloaded), merged into runs and scored by whole-clip audio-text similarity — not word-level ASR.
- Every result carries a `confident` flag and `warnings`. **Ear-checking the output is the real ground truth this pipeline serves** — it narrows a full recording down to a short, mostly-correct list of candidates to verify, not a replacement for checking them.
- Per-ayah pacing (`pace_hints`) is auto-downloaded and cached from a companion dataset on first use, so duration estimates are calibrated per-ayah rather than one flat rate for everything.

The code itself is intentionally comment-light — see the arguments table above and the module-level docstrings at the top of `ayah_aligner.py`/`boundary_refine.py` for how to use it; read the source directly before changing scoring logic.

## 📄 Licenses

| Component                                                            | License                                  |
| -------------------------------------------------------------------- | ---------------------------------------- |
| This code                                                            | MIT                                      |
| [tadabur-embedding](https://huggingface.co/FaisaI/tadabur-embedding) | CC BY-NC 4.0                             |
| [quran-align](https://github.com/cpfair/quran-align) timestamps      | CC BY 4.0                                |
| Reference audio ([everyayah.com](https://everyayah.com))             | fetched at build time, not redistributed |

Part of the tadabur family: [dataset](https://huggingface.co/datasets/FaisaI/tadabur) · [embedding](https://huggingface.co/FaisaI/tadabur-embedding) · [align](https://github.com/FaisaI/tadabur-align) · this cropper.
