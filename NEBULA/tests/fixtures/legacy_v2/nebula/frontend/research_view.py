"""Read the public experiment journals without loading ML libraries."""
import json
from pathlib import Path


def journal_rows(path):
    if not path.exists():
        return []
    if path.stat().st_size > 4_000_000:
        raise ValueError("Research journal is larger than the supported local experiment")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The writer may still be appending its last line.
            continue
    return result


def research_payload(log_dir, config_dir, scenario):
    logs, configs = Path(log_dir), Path(config_dir)
    manifest_path = configs / "research_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    configurations = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(configs.glob("participant_*.json"))]
    trainers = [p for p in configurations if p["device_args"]["role"] == "trainer"]
    privacy, evaluations, errors = [], [], []
    for p in configurations:
        idx = p["device_args"]["idx"]
        privacy.extend(journal_rows(logs / f"participant_{idx}_privacy.jsonl"))
        evaluations.extend(journal_rows(logs / f"participant_{idx}_evaluation.jsonl"))
        errors.extend(journal_rows(logs / f"participant_{idx}_privacy_error.jsonl"))
    privacy.sort(key=lambda r: (r["round"], r["participant"]))
    evaluations.sort(key=lambda r: r["round"])
    config = trainers[0] if trainers else (configurations[0] if configurations else {})
    dp = config.get("training_args", {}).get("dp", {})
    expected = {p["device_args"]["idx"] for p in trainers}
    rounds = int(config.get("scenario_args", {}).get("rounds", 0))
    last = {r["participant"]: r for r in privacy}
    complete_accounting = bool(expected) and all(
        i in last and last[i].get("complete") and last[i]["round"] == rounds for i in expected)
    latest_epsilon = max((r["epsilon"] for r in last.values()), default=None)
    is_v2 = manifest is not None and manifest.get("protocol") == "project2-public-evaluation-v2"
    return {
        "scenario": dict(scenario), "protocol": manifest.get("protocol") if manifest else "legacy",
        "execution": manifest.get("execution", "NEBULA container scenario") if manifest else "Legacy scenario",
        "manifest": manifest, "dp": dp, "privacy": privacy, "evaluations": evaluations,
        "errors": errors, "accounting_complete": complete_accounting,
        "epsilon_max": latest_epsilon if dp.get("enabled") else None,
        "expected_clients": sorted(expected), "rounds": rounds,
        "final_test": next((r["final_test"] for r in reversed(evaluations) if "final_test" in r), None),
        "comparison_key": {"protocol": manifest.get("protocol") if manifest else "legacy",
                           "execution": manifest.get("execution", "NEBULA container scenario") if manifest else "Legacy scenario",
                           "seed": manifest.get("seed") if manifest else None,
                           "iid": manifest.get("iid") if manifest else None,
                           "alpha": manifest.get("dirichlet_alpha") if manifest else None,
                           "rounds": rounds, "epochs": config.get("training_args", {}).get("epochs"),
                           "model": config.get("model_args", {}).get("model"),
                           "clients": manifest.get("clients") if manifest else None},
        "scope": "Conditional record-level training accounting with fixed public synthetic allocation and sizes. No hospital-level guarantee or clinical validation.",
        "rng": "CSPRNG requested" if dp.get("secure_mode") else "Experimental RNG, not cryptographically secure",
        "interpretation": "No DP bound" if not dp.get("enabled") else ("Complete training accounting" if complete_accounting and not errors and is_v2 else "Pending/incomplete training accounting"),
    }
