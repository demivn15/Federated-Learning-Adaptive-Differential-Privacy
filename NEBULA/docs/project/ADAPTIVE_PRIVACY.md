# Protocol and adaptive clipping

The active protocol is `project2-public-evaluation-v3`. It evaluates one aggregation server and three trainers with WDBC data, a fixed public split and public synthetic allocation. The data labels are 0 malignant and 1 benign.

## Data flow

The 569 records form three disjoint subsets: 364 for trainer partitions, 91 for public calibration/validation and 114 for final testing. The StandardScaler fits only the public subset. Each trainer reads its assigned training partition. The server receives a public placeholder dataset and does not perform private local training.

The default Non-IID allocation is Dirichlet alpha 0.5 with seed 42. It produces center sizes 17, 55 and 292. Class skew and quantity skew both vary in this demonstration. Publishing the synthetic allocation and sizes is part of the experimental design, outside the training privacy claim.

## Mechanism

Before the first update, the privacy plugin calibrates a constant sigma to the configured epsilon ceiling and delta, according to the public sampling plan and the total planned optimizer steps. Both fixed and adaptive modes follow this same calibration.

At the beginning of an adaptive round, `PublicAdaptiveClipper` evaluates per-record cross-entropy gradient norms on the public reference set at the incoming model. It calculates a quantile, bounds it and smooths the result:

```text
target = clamp(quantile(reference norms), minimum C, maximum C)
current C = (1 - smoothing) * previous C + smoothing * target
```

The default median is 0.5. The new target receives weight 0.5. The initial C is 1, with lower and upper bounds 0.1 and 5. The controller performs no optimizer steps on the reference records and does not change training gradients while measuring the reference norms.

The initial model is independent of private records. Later reference calculations are post-processing of the previous private model history with public data. Private gradient norms, private losses and final-test records do not enter the adaptation decision.

Opacus clips private individual gradients at the selected C and adds Gaussian noise with standard deviation sigma times C to their sum before normalization. C and the noise scale change together. The noise-to-clipping ratio stays sigma, so the conditional RDP cost follows the fixed sampling plan and step count. The accountant persists across all rounds for each client.

## Accounting guards

- Rounds must be consecutive and share one persistent accountant.
- Existing non-empty privacy journals cannot be reset by restarting the same scenario.
- A changed sampling plan requires a new scenario and calibration.
- An unexpected optimizer step or unapproved change to sigma/C is rejected.
- An adaptive run requires an explicitly designated public reference set.
- A failed private round or startup produces evidence instead of a complete privacy claim.

This is conditional record-level training accounting, not hospital-level DP or a complete deployment guarantee. Secure randomness remains optional and explicitly identified. The default experimental configuration does not enable `torchcsprng`.

## Evaluation and interpretation

The public reference set serves both adaptation and validation. Its scores are feedback measurements, not independent estimates of generalization. The final-test results have already been observed during exploratory work. A publication needs separate parameter selection and assessment, repeated runs, and a precise neighboring-dataset definition consistent with the accountant.

The current mechanism is an experimental public-reference controller. It does not reproduce the private user-level quantile estimator in Andrew et al. and does not establish algorithmic novelty. The intended comparison is fixed versus adaptive clipping under comparable epsilon/delta bounds, model, data, sampling plan, rounds and local epochs.

## Source map

| Source | Role |
| --- | --- |
| `nebula/core/training/adaptive_clipping.py` | Public-gradient quantile and bounded smoothing |
| `nebula/core/training/privacy_policy.py` | Parameter validation and noise calibration |
| `nebula/core/training/dp.py` | Opacus integration, persistent accounting and guards |
| `nebula/core/training/lightning_dp.py` | NEBULA Lightning training bridge |
| `nebula/core/datasets/breast_cancer/research_data.py` | Disjoint split, scaling and synthetic partitions |
| `nebula/core/training/research_results.py` | Global evaluation journals |
| `nebula/frontend/research_view.py` | Evidence payloads |
| `nebula/frontend/static/js/research.js` | Charts, comparisons and exports |

References appear in the main README and in the preserved historical development notes under `archive/development/`.
