"""Fixed public-data benchmark: 364 training, 91 calibration, 114 test records.

Synthetic hospital allocation is a public experimental design. The accountant
conditions on that fixed allocation and public sizes. It does not protect the
allocation mechanism or claim hospital-level/production privacy.
"""
import json
from pathlib import Path

import numpy as np


def prepare_arrays(iid=False, alpha=0.5, seed=42, clients=3):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    dataset = load_breast_cancer()
    all_ids = np.arange(len(dataset.target))
    pool, test = train_test_split(all_ids, test_size=0.2, random_state=seed,
                                 stratify=dataset.target)
    train, public = train_test_split(pool, test_size=0.2, random_state=seed,
                                    stratify=dataset.target[pool])
    scaler = StandardScaler().fit(dataset.data[public])
    x = scaler.transform(dataset.data).astype(np.float32)
    y = np.asarray(dataset.target, dtype=np.int64)  # 0 malignant, 1 benign
    rng = np.random.default_rng(seed)
    if iid:
        parts = [[] for _ in range(clients)]
        for label in (0, 1):
            ids = train[y[train] == label].copy()
            rng.shuffle(ids)
            for k, chunk in enumerate(np.array_split(ids, clients)):
                parts[k].extend(chunk.tolist())
    else:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        for attempt in range(1000):
            parts = [[] for _ in range(clients)]
            for label in (0, 1):
                ids = train[y[train] == label].copy()
                rng.shuffle(ids)
                counts = rng.multinomial(len(ids), rng.dirichlet([alpha] * clients))
                for k, chunk in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
                    parts[k].extend(chunk.tolist())
            if min(map(len, parts)) >= 16:
                break
        else:
            raise ValueError("Could not allocate at least 16 training records per center")
    flat = [i for part in parts for i in part]
    if len(flat) != len(set(flat)) or set(flat) != set(train):
        raise ValueError("Training partitions must be disjoint and cover the training pool")
    if set(train) & set(public) or set(train) & set(test) or set(public) & set(test):
        raise ValueError("Training, public calibration and final test must be disjoint")
    manifest = {
        "protocol": "project2-public-evaluation-v3", "dataset": "WDBC",
        "privacy_policy": "constant-sigma-public-reference-adaptive-clipping-v1",
        "public_reference_purpose": "Normalization, validation and adaptive clipping. Final test excluded.",
        "seed": seed, "label_mapping": {"0": "malignant", "1": "benign"},
        "training_records": len(train), "public_validation_records": len(public),
        "final_test_records": len(test), "iid": bool(iid), "dirichlet_alpha": alpha,
        "clients": [{"participant": k + 1, "records": len(ids),
                     "malignant": int(np.sum(y[ids] == 0)),
                     "benign": int(np.sum(y[ids] == 1))} for k, ids in enumerate(parts)],
        "privacy_scope": "Record-level training accounting conditional on public fixed synthetic allocation and sizes. No hospital-level or full deployment guarantee.",
        "normalization": "StandardScaler fitted only on the 91 designated public calibration records",
    }
    return x, y, parts, public, test, manifest


def write_research_partitions(dataset):
    import h5py
    root = Path(dataset.config_dir)
    participants = sorted(root.glob("participant_*.json"))
    configs = [json.loads(p.read_text(encoding="utf-8")) for p in participants]
    trainers = [c for c in configs if c["device_args"]["role"] == "trainer"]
    servers = [c for c in configs if c["device_args"]["role"] == "server"]
    if len(servers) != 1 or len(trainers) != 3 or len(configs) != 4:
        raise ValueError("Project 2 requires one server and three trainers")
    if not dataset.iid and dataset.partition != "dirichlet":
        raise ValueError("Project 2 Non-IID requires Dirichlet, not Percentage")
    x, y, parts, public, test, manifest = prepare_arrays(
        iid=dataset.iid, alpha=dataset.partition_parameter, seed=dataset.seed)
    def write(path, prefix, ids):
        with h5py.File(root / path, "w") as stream:
            item = stream.create_dataset(prefix + "_data", data=x[ids], compression="lzf")
            item.attrs["__type__"] = "array"
            item.attrs["num_classes"] = 2
            stream.create_dataset(prefix + "_targets", data=y[ids])
    for config, ids, row in zip(trainers, parts, manifest["clients"]):
        idx = config["device_args"]["idx"]
        row["participant"] = idx
        write(f"participant_{idx}_train.h5", "train", ids)
    # Server never trains. Public records satisfy NEBULA's dataset interface.
    write(f"participant_{servers[0]['device_args']['idx']}_train.h5", "train", public)
    write("global_test.h5", "test", test)
    write("public_validation.h5", "test", public)
    (root / "research_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
