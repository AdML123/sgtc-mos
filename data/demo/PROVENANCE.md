# Demo dataset provenance

The 20 wav clips in `wavs/` (10 train, 10 test) and the cached encoder
features in `features/` are a smoke-test subset of the **publicly distributable
portion of the VoiceMOS Challenge 2022 main-track release** (samples from the
Voice Conversion Challenges and ESPnet-TTS).

- Source archive: Zenodo record [6572573](https://zenodo.org/records/6572573),
  `main.tar.gz`, SHA-256-verified via the record's MD5
  (`fc880c2a208c3285a47bd9a64f34eb11`).
- Utterance IDs are prefixed with their split (`train_*` / `test_*`) and MOS
  labels are in `train_mos.csv` / `test_mos.csv`.
- Licenses of the redistributed samples are copied verbatim under `licenses/`.
- The full corpus is NOT redistributed here; fetch it from the Zenodo record
  above and follow the repository README to reproduce the paper tables.
- Features were extracted with `facebook/wav2vec2-base-960h` (half precision,
  CPU-decoded to fp16 `.npy`), exactly as `sgtc/extract_features.py` does.

This subset exists only so that a fresh clone can run the pipeline end to end
without downloading any restricted or large resource
(`python -m sgtc.run_demo`).
