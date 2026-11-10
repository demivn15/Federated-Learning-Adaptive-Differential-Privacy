# Adaptive Differential Privacy with a Public Reference Set

The current experiment compares non-private federated training, fixed DP-SGD and DP-SGD with adaptive clipping. It replaces the precomputed decreasing-noise option for new scenarios. Existing scheduled-noise results retain their original mode and protocol.

## What adapts

The clipping limit C responds to the current model. At the start of each local round, the controller evaluates individual cross-entropy gradient norms on the 91 designated public calibration records. It takes a configurable quantile, bounds the result, and smooths it with the previous C:

```text
target = clamp(quantile(public gradient norms), C_min, C_max)
C_round = (1 - weight) * C_previous + weight * target
```

Defaults are median (quantile 0.5), weight 0.5, initial C = 1, C_min = 0.1 and C_max = 5. Public data and the incoming model determine the decision. The controller does not inspect private training gradient norms, private losses, or final-test records. It does not perform optimizer updates on the public reference data.

The incoming model starts from data-independent initialization. Later rounds receive the global model aggregated from earlier DP updates. The reference calculations are post-processing of this DP history together with public data. This is **public-reference adaptive clipping**. It is not a reproduction of a method which privately estimates quantiles of individual clients' private gradients, and it does not establish a new algorithmic contribution.

## Why sigma remains constant

Opacus calibrates a constant noise multiplier sigma before training from the public sampling plan, number of rounds and local steps, epsilon ceiling and delta. Each optimizer step clips individual gradients at C_round and adds Gaussian noise with standard deviation `sigma * C_round` to their sum before normalization.

Changing C therefore changes the clipping sensitivity and noise scale together. Their ratio remains sigma. With fixed sampling and step counts, each conditional mechanism has the same RDP cost regardless of the selected C. The persistent per-client accountant composes those costs across rounds. The selected C only depends on the previous DP history and public records, so this controller does not require a separate private-statistics query. A controller consulting unprotected private gradient norms would require a different analysis and additional protection.

The scope remains conditional record-level training accounting with a fixed, public synthetic allocation and public partition sizes. It does not cover hospital-level participation, the label-dependent allocation procedure or a complete medical deployment. Experimental RNG remains explicitly identified. Secure RNG still requires a compatible torchcsprng installation. Releasing multiple models trained on the same real private records also requires accounting across releases.

## What stays fixed for a comparison

The benchmark retains the 364 training, 91 public calibration/validation and 114 final-test records, three simulated centers, existing Non-IID allocation, MLP, FedAvg and Poisson sampling. This update does not change the data distribution, architecture or number of rounds to improve the reported accuracy. Those are separate future experiments.

All new scenarios carry protocol `project2-public-evaluation-v3`. The results panel can still read v2 evidence but marks v2-versus-v3 comparisons as different protocols. Create fresh baseline, fixed and adaptive scenarios for the v3 comparison. Adaptive mode intentionally changes C, so match the initial C and compare the realized values as well as the effective epsilon bounds.

## Install into the existing Ubuntu project

1. In the terminal running NEBULA, press Ctrl+C and wait for its containers to stop.
2. In Ubuntu, run this command:

```bash
bash "/mnt/c/Users/USER/Documents/Codex/2026-09-28/ver-s-la-idea-que-yo/outputs/project2-nebula/start_adaptive_wsl.sh"
```

The script applies only reviewed source changes, runs the tests in the installed environment and starts NEBULA. The updater accepts the known v1/v2 source hashes, rejects unexpected local edits, and backs up changed sources under `project2-backups`. It refuses to apply while affected containers run. It preserves saved experiments, databases, datasets, environments and images. No new dependency or image rebuild is required in experimental RNG mode.

If a test fails, the script stops before platform startup. Review that output rather than launching a private experiment with incomplete checks.

3. Visit `http://localhost:6060` and refresh the deployment page with Ctrl+F5.
4. Create three new scenarios named, for example, `WDBC_V3_BASE_01`, `WDBC_V3_FIJA_01`, and `WDBC_V3_ADAPTATIVA_01`.

For the initial functional comparison, retain CFL, four total nodes, node 0 as server with Start enabled, three trainers, Star, BreastCancer, Non-IID, Dirichlet alpha 0.5, MLP, FedAvg, CPU, three rounds and one epoch. Disable attacks and unrelated defenses. Select the adaptive option only for the third scenario. Both private modes start with epsilon 8, delta 0.00001 and C = 1. Leave the adaptive controls at their documented defaults for the first run.

## What the results page shows

- Accumulated epsilon and actual sigma per client.
- C selected in each round and the resulting `sigma * C` noise scale.
- An adaptation table containing the previous C, observed public quantile, bounded target, selected C and number of reference records.
- Public-validation curves and a separate final-test confusion matrix.
- CSV/JSON exports with mode, applied clipping limit, noise scale and adaptation evidence.

Constant sigma is expected in adaptive mode. The C and sigma-times-C plots show the adaptation. C can increase, decrease, or stay unchanged depending on the public reference gradients. None of those directions guarantees better classification.

The 91 records now serve both validation and adaptation. They are a public reference set, not an independent assessment of generalization. Final-test observations must not guide later tuning. The existing test has already been inspected in exploratory work, so a publication needs a frozen evaluation protocol with appropriately separated model selection and assessment.

## Verification status

The Windows checks passed six privacy tests and skipped two environment-specific tests for HDF5 and NEBULA Lightning. All eight privacy tests subsequently passed in Ubuntu, including actual two-round NEBULA Lightning adaptive training and HDF5 partition writing. The user's terminal output confirmed the result and the shared Lightning privacy journal records both completed adaptive rounds. The release suite also checks migration, Docker startup failures, scenario-scoped cleanup, queue ordering and the actual stop endpoint with a mocked Docker client. Behavioral frontend checks cover the controls, historical labels, comparison rules and exported CSV rows.

A measured standalone CPU run completed baseline, fixed and adaptive training for three rounds. It is separate from a complete NEBULA container experiment.

In that adaptive smoke run, client 1 selected C values 1.484127, 1.661248 and 1.593120. Its sigma stayed 1.013250. The maximum final epsilon across clients was 7.941367. Fixed and adaptive modes produced identical privacy-accountant histories for the same sampling plans. This verifies the intended mechanism and accounting path under the stated assumptions, not statistical superiority or clinical validity.

The separate updater and frontend checks validate migration safety and presentation behavior. The platform restarted successfully and its live deployment page exposes the adaptive controls. Three new v3 NEBULA container scenarios were submitted sequentially. Their completion and measured results are recorded separately under `results/privacy_v3_nebula_20261104`, rather than inferred from the standalone smoke run. The chat cannot access the WSL service directly, but the local authenticated NEBULA API supports submission and verification after startup.

## Measured NEBULA model results

Each condition trained for the planned three rounds and produced a final evaluation on 114 test records. Both private conditions completed nine privacy-journal entries, three clients times three rounds, with maximum epsilon 7.941367 at delta 0.00001.

| Scenario | Accuracy | Malignant recall | Malignant F1 | Malignant cases detected |
| --- | ---: | ---: | ---: | ---: |
| WDBC_V3_BASE_01 | 94.7% | 97.6% | 93.2% | 41 of 42 |
| WDBC_V3_FIJA_02 | 82.5% | 52.4% | 68.8% | 22 of 42 |
| WDBC_V3_ADAPTATIVA_01 | 83.3% | 54.8% | 70.8% | 23 of 42 |

In the adaptive NEBULA run, client 1 selected C = 1.512700, 1.700347 and 1.782626. Sigma stayed 1.013250. These are the container-run values, distinct from the standalone smoke values above.

These are single exploratory runs, not means or evidence of statistical superiority. The adaptive run detected one additional malignant record relative to the fixed run. The non-private run detected substantially more malignant records. No parameter tuning followed observation of these final-test results.

## Operational failure and cleanup correction

The first v3 fixed attempt, WDBC_V3_FIJA_01, failed during Docker startup with 409 container-name conflicts. It registered no nodes and produced no training metrics. Its evidence is preserved as a failed attempt and it is excluded from model-quality comparisons. WDBC_V3_FIJA_02 repeated the same training configuration after the startup correction.

The launcher now propagates creation/start errors instead of attempting to start an unassigned container identifier. The queue waits for cleanup before reusing names. Cleanup matches the specific scenario, so a delayed request cannot remove the next scenario's participants. Startup failures receive a failed status and an error journal. Completed training evidence persists in the bind-mounted result directories during cleanup.

A missing Docker import in the cleanup adjustment required a final controller correction after the fixed model had completed training. The correction adds a regression test which executes the actual stop-endpoint body with mocked Docker dependencies. Model metrics and privacy journals were saved before restarting the platform, without repeating or selecting a different training result.

The final Ubuntu check passed all eight privacy tests and all nine release/lifecycle tests. An actual request to the corrected stop endpoint returned HTTP 200. The baseline and adaptive scenarios report `completed`, and the fixed scenario reports NEBULA's terminal `finished` state. All three have the required final training evidence. The consolidated validation record is `privacy_adaptive_validation.json`.

## Scientific references

- Andrew, G., Thakkar, O., McMahan, H. B., and Ramaswamy, S. (2021). [Differentially Private Learning with Adaptive Clipping](https://papers.nips.cc/paper_files/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html). NeurIPS. This motivates adaptive clipping, but its user-level private-quantile method differs from this public-reference controller.
- [Opacus DPOptimizer API](https://opacus.ai/api/optim/dp_optimizer.html). Documents individual-gradient clipping and Gaussian noise proportional to sigma times C.
- [Opacus PrivacyEngine API](https://opacus.ai/api/privacy_engine.html). Documents the persistent training accountant and secure-randomness option.
