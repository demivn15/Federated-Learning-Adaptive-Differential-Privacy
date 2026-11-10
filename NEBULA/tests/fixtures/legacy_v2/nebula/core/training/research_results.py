"""Small JSONL records for the live monitor. Never serialize private examples."""
import json
import time
from pathlib import Path


def append_record(config, kind, row):
    participant = config.participant if hasattr(config, "participant") else config
    root = Path(participant["tracking_args"]["log_dir"]) / participant["scenario_args"]["name"]
    root.mkdir(parents=True, exist_ok=True)
    idx = participant["device_args"]["idx"]
    record = {"schema": 2, "kind": kind, "participant": idx, "timestamp": time.time(), **row}
    with (root / f"participant_{idx}_{kind}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def evaluate_public(model, dataset):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
    from torch.utils.data import DataLoader
    previous = model.training
    device = next(model.parameters()).device
    probabilities, labels = [], []
    try:
        model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=32, shuffle=False):
                probabilities.extend(torch.softmax(model(x.to(device)), dim=1)[:, 0].cpu().tolist())
                labels.extend(y.tolist())
    finally:
        model.train(previous)
    scores = np.asarray(probabilities)
    predicted = np.where(scores >= 0.5, 0, 1)
    return {"records": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
            "malignant_recall": float(recall_score(labels, predicted, pos_label=0, zero_division=0)),
            "malignant_f1": float(f1_score(labels, predicted, pos_label=0, zero_division=0)),
            "roc_auc": float(roc_auc_score(np.asarray(labels) == 0, scores)) if len(set(labels)) == 2 else None,
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            "class_order": ["malignant", "benign"]}


def record_server_round(trainer, round_index, elapsed, final=False):
    config = trainer.config
    p = config.participant
    if p["device_args"]["role"] != "server" or not getattr(trainer.datamodule, "public_validation", None):
        return
    row = {"round": round_index + 1, "round_seconds": elapsed,
           "validation": evaluate_public(trainer.model, trainer.datamodule.public_validation)}
    if final:
        row["final_test"] = evaluate_public(trainer.model, trainer.datamodule.test_set)
    append_record(config, "evaluation", row)
