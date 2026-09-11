# Neuron-Merged SSL Scoring (NMSS)

Reference implementation for **"Halving Feed-Forward Neurons Improves Self-Supervised Speech Quality Assessment"**.

## What this reproduces

The manuscript studies where inference cost should be removed from a frozen
self-supervised encoder used for mean-opinion-score (MOS) prediction:

1. **Axis diagnosis** (`sgtc/diagnosis/`) — permutation tests showing the pooled
   scoring head is exactly invariant to temporal reordering while channel
   reordering collapses prediction, at every encoder layer.
2. **Neuron merging surgery** (`sgtc/compression/channel_pruning.py`) — k-means
   clustering of every feed-forward block's intermediate neurons, merged to
   1/2, 1/3, or 1/4 width; only a ridge head is retrained.
3. **Temporal compression controls** (`sgtc/compression/`) — adjacent merging
   with count-weighted pooling (mean-preserving), ToMe-style global merging,
   random dropping, and merge-priority ablations.
4. **Cross-domain and cross-backbone evaluation** (`sgtc/evaluation/run_cross.py`).

Headline result on the VoiceMOS Challenge 2022 distributable corpus
(2,254 train / 741 test utterances, wav2vec2-base-960h frozen):

| system | LCC | SRCC | RMSE |
|---|---|---|---|
| uncompressed | 0.668 | 0.656 | 0.830 |
| neuron merging, FFN ×1/3 | **0.820** | **0.826** | **0.566** |
| magnitude pruning, FFN ×1/3 | 0.767 | 0.771 | 0.657 |
| temporal merging (weighted), T×1/2 | 0.651 | 0.641 | 0.955 |

All numbers in the paper are generated from `results/` by
`python -m sgtc.fill_numbers` (single source of truth; no hand copying).

## Requirements

- Python 3.10, PyTorch >= 2.5 with a CUDA 12.8 capable GPU (tested on
  RTX 5060 Ti, Blackwell sm_120), transformers, librosa, scikit-learn, scipy,
  pandas, matplotlib. See `environments/requirements.lock.txt` for exact pins.
- Datasets (not redistributed here):
  - **VoiceMOS Challenge 2022 main track** — Zenodo record
    [6572573](https://zenodo.org/records/6572573) (`main.tar.gz`, md5
    `fc880c2a208c3285a47bd9a64f34eb11`). Place the distributable wavs and
    per-split MOS lists under `data/voicemos2022/`.
  - **NISQA Corpus** (cross-domain test batteries) — see the NISQA release
    (Mittag et al., Interspeech 2021) for access; place `NISQA_TEST_*` folders
    and `NISQA_corpus_file.csv` under `data/nisqa/`.
- Encoder weights from the Hugging Face Hub: `facebook/wav2vec2-base-960h`,
  `facebook/hubert-base-ls960`, `microsoft/wavlm-base` (local copies under
  `models/`, see `sgtc/config.py`).

## Reproduction pipeline

```bash
python -m pytest                       # unit tests for metrics/compression
python -m sgtc.run_extract             # stage A: cache wav2vec2 features
python -m sgtc.train_regression_head   # (imported by runners) ridge heads
python -m sgtc.diagnosis.shuffle_test  # via runners below
python -m sgtc.run_surgery             # stage C: prune/merge surgery + re-extract
python -m sgtc.run_deploy              # in-encoder temporal variant + timing
python -m sgtc.run_attention_scores    # attention-priority ablation inputs
python -m sgtc.evaluation.run_experiments   # Table I
python -m sgtc.evaluation.run_ablations     # Table II
python -m sgtc.evaluation.run_cross         # Table III + cross-backbone
python -m sgtc.diagnosis.layer_analysis     # Fig. 4 data
python -m sgtc.fill_numbers                 # paper numbers.tex
```

Result tables land in `results/tables/`, per-utterance predictions in
`results/preds/`, figures in `paper/figs/` (via `sgtc/visualization/`).

## Repository layout

```
sgtc/            experiment code (config, extraction, surgery, compression,
                 diagnosis, evaluation, visualization, fill_numbers, linter)
environments/    pinned requirements
results/         tables (CSV/JSON) and per-utterance predictions produced by
                 the pipeline above; the exact artifacts backing the paper
tests via sgtc/  pytest unit tests next to each module
data/            NOT included — see Requirements for official sources
models/          NOT included — fetch from the Hugging Face Hub
```

## Revision additions (v0.2.0)

- `sgtc/run_seed_stability.py` — five-seed k-means stability (Table I plus-minus std, Fig. 3 band)
- `sgtc/evaluation/run_revision_stats.py` — merge-vs-prune paired tests, bootstrap CI, NISQA calibration
- `sgtc/evaluation/run_mlp_head.py` — MLP-head control, five seeds
- `sgtc/evaluation/run_size_ablation.py` — training-size ablation (regularization evidence)
- `sgtc/run_gpu_tail.py` and `sgtc/fix_merge2_pair.py` — wall-clock timing and cache-consistency tooling

## Boundaries

- The labeled corpus is the officially distributable portion of the challenge
  data, smaller than the full benchmark; absolute correlations are not
  comparable with challenge leaderboards.
- In-encoder temporal merging shows no wall-clock gain at batch size one with
  short utterances (reported honestly in the manuscript).

## License

Code: MIT. Derived result tables: CC-BY-4.0. Dataset licenses remain with
their providers and the data is not redistributed.
