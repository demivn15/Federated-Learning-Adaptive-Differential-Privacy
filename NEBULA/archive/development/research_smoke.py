"""Three measured standalone smoke runs. These are NOT NEBULA container runs."""
import argparse
import copy
import json
import secrets
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "nebula"))
import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus.data_loader import DPDataLoader
from nebula.core.datasets.breast_cancer.research_data import prepare_arrays
from nebula.core.training.dp import DifferentialPrivacyPlugin, SimpleDPState
from nebula.core.training.research_results import append_record, evaluate_public
from nebula.core.training.privacy_policy import validate_dp_config


def model():
    return torch.nn.Sequential(torch.nn.Linear(30, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2))


def run(output, rounds=3):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError("Choose a new smoke-test output directory")
    output.mkdir(parents=True)
    x, y, partitions, public, test, manifest = prepare_arrays()
    manifest["execution"] = "Standalone CPU integration smoke test, not a NEBULA network/container run"
    tensor_x, tensor_y = torch.tensor(x), torch.tensor(y)
    public_ds, test_ds = TensorDataset(tensor_x[public], tensor_y[public]), TensorDataset(tensor_x[test], tensor_y[test])
    with torch.random.fork_rng():
        torch.manual_seed(42)
        initial = copy.deepcopy(model().state_dict())
    summaries = []
    for mode in ("baseline", "fixed", "adaptive"):
        name = "smoke_" + mode
        config_dir, log_dir = output / "config" / name, output / "logs" / name
        config_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)
        (config_dir / "research_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        configs, clients = [], []
        for idx in range(4):
            p = {"device_args": {"idx": idx, "role": "server" if idx == 0 else "trainer"},
                 "scenario_args": {"name": name, "rounds": rounds},
                 "tracking_args": {"log_dir": str(output / "logs")},
                 "model_args": {"model": "MLP"},
                 "training_args": {"epochs": 1, "dp": validate_dp_config({
                     "enabled": mode != "baseline", "mode": "adaptive" if mode == "adaptive" else "fixed",
                     "target_epsilon": 8.0, "noise_end": 0.7})}}
            (config_dir / f"participant_{idx}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
            configs.append(SimpleNamespace(participant=p))
            if idx:
                net = model()
                net.load_state_dict(initial)
                data = TensorDataset(tensor_x[partitions[idx-1]], tensor_y[partitions[idx-1]])
                plugin = None if mode == "baseline" else DifferentialPrivacyPlugin(
                    **{k:v for k,v in p["training_args"]["dp"].items() if k != "enabled"},
                    total_rounds=rounds, config=configs[-1])
                clients.append((net, data, plugin))
        global_state = copy.deepcopy(initial)
        global_model = model()
        started = time.perf_counter()
        for r in range(rounds):
            round_started = time.perf_counter()
            states = []
            for net, data, plugin in clients:
                net.load_state_dict(global_state)
                optimizer = torch.optim.Adam(net.parameters(), lr=.001)
                loader = DataLoader(data, batch_size=8, shuffle=True, drop_last=False,
                                    generator=torch.Generator().manual_seed(secrets.randbits(63)))
                network = net
                state = SimpleDPState()
                if plugin:
                    state.extras.update(dataloader=loader, epochs=1, round_index=r)
                    state.extras.update(public_reference_dataset=public_ds, public_reference_is_public=True)
                    plugin.on_train_start(net, optimizer, state)
                    loader, network, optimizer = (state.extras[k] for k in ("dataloader", "model", "optimizer"))
                else:
                    loader = DPDataLoader.from_data_loader(loader)
                network.train()
                for inputs, labels in loader:
                    optimizer.zero_grad()
                    logits = network(inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels) if len(labels) else logits.sum()
                    loss.backward()
                    optimizer.step()
                if plugin:
                    plugin.on_train_end(state)
                states.append(copy.deepcopy(net.state_dict()))
            total = sum(len(data) for _, data, _ in clients)
            global_state = {key: sum(len(data)/total * state[key] for (_, data, _), state in zip(clients, states)) for key in initial}
            global_model.load_state_dict(global_state)
            row = {"round": r+1, "round_seconds": time.perf_counter()-round_started,
                   "validation": evaluate_public(global_model, public_ds)}
            if r == rounds-1:
                row["final_test"] = evaluate_public(global_model, test_ds)
            append_record(configs[0], "evaluation", row)
            print(json.dumps({"mode": mode, "round": r+1, "public_accuracy": row["validation"]["accuracy"]}), flush=True)
        summary = {"mode": mode, "execution": manifest["execution"], "rounds": rounds,
                   "wall_seconds": time.perf_counter()-started, "final_test": row["final_test"],
                   "epsilon_max": max((c[2]._privacy_engine.get_epsilon(delta=1e-5) for c in clients if c[2]), default=None)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.rounds)
