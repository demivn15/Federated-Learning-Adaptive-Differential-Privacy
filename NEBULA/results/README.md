# Recorded experimental evidence

| Directory | Execution source |
| --- | --- |
| `nebula_v3_20261104/` | Full NEBULA container runs, with separate baseline, fixed and adaptive evaluations |
| `smoke_v3_20261104/` | Independent CPU integration checks, not container/network experiments |
| `history/nebula_v2_20261104/` | Earlier exported fixed/scheduled-noise NEBULA evidence |
| `history/smoke_v2_20261104/` | Earlier standalone fixed/scheduled-noise checks |
| `history/pilot_*` | Initial feasibility pilots with a different label mapping |

Preserved results retain their original contents, including execution timestamps, configuration paths and protocol labels. They are not silently relabelled as adaptive runs. The original pilot explicitly mapped 0 to benign and 1 to malignant. The current v3 protocol follows scikit-learn's 0 malignant and 1 benign mapping.

The v3 fixed attempt ending in `_15_08_40` failed at Docker startup and produced no training metrics. Its failed evidence remains available. The successful fixed retry ending in `_15_25_30` supplies the fixed-DP model results. A cleanup correction required a platform restart after that model had completed its planned training and accounting. Its pre-cleanup evidence was preserved too.

New experiments should write to `results/generated/`, which is ignored by Git. Reviewed exports can later be placed in a dated directory with their execution source and conditions documented. Nine client-round rows represent one three-client, three-round run, not nine independent experiments.
