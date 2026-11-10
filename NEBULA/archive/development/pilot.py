"""Small CPU feasibility pilot. This script does not run NEBULA containers.

Public dataset: UCI Breast Cancer Wisconsin (Diagnostic), DOI 10.24432/C5DW2B.
Labels are explicitly mapped to 0=benign and 1=malignant.
Only public calibration/test metrics are logged. Client models are aggregated.
"""
import argparse
import copy
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import secrets
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def model():
    # Same layer dimensions as NEBULA's BreastCancerModelMLP at the pinned commit.
    return nn.Sequential(nn.Linear(30, 256), nn.ReLU(), nn.Linear(256, 128),
                         nn.ReLU(), nn.Linear(128, 2))


def prepare_data(seed):
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float32)
    y = (np.asarray(dataset.target) == 0).astype(np.int64)  # malignant is positive
    pool, test = train_test_split(np.arange(len(y)), test_size=0.2,
                                 stratify=y, random_state=seed)
    private, public = train_test_split(pool, test_size=0.2,
                                      stratify=y[pool], random_state=seed + 1)
    # Fit exclusively on the designated public calibration data.
    scaler = StandardScaler().fit(x[public])
    transformed = torch.tensor(scaler.transform(x), dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.long)
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(3)]
    for label in (0, 1):
        records = rng.permutation(private[y[private] == label])
        for client, part in zip(clients, np.array_split(records, 3)):
            client.extend(part.tolist())
    joined = [i for client in clients for i in client]
    assert len(joined) == len(set(joined)) == len(private)
    assert set(joined) == set(private)
    assert set(private).isdisjoint(public) and set(private).isdisjoint(test)
    assert set(public).isdisjoint(test)
    assert set(private) | set(public) | set(test) == set(range(len(y)))
    assert all(len(np.unique(y[client])) == 2 for client in clients)
    manifest = {
        'dataset': 'UCI Breast Cancer Wisconsin (Diagnostic)',
        'dataset_doi': '10.24432/C5DW2B',
        'total_records': len(y), 'features': int(x.shape[1]),
        'labels': {'0': 'benign', '1': 'malignant'},
        'public_calibration_indices': public.tolist(),
        'public_test_indices': test.tolist(),
        'simulated_client_indices': clients,
        'private_training_records': len(private),
        'public_calibration_records': len(public),
        'public_test_records': len(test),
        'client_sizes': [len(c) for c in clients],
        'scaler_fit_source': 'designated public calibration split only',
        'partition_and_initialization_seed': seed,
        'sampling_and_noise_seeds': 'independent, not exported',
    }
    return transformed, targets, clients, public, test, manifest


def public_metrics(network, x, y, indices):
    network.eval()
    with torch.no_grad():
        score = network(x[indices]).softmax(dim=1)[:, 1].numpy()
    actual = y[indices].numpy()
    prediction = (score >= 0.5).astype(np.int64)
    return {
        'accuracy': float(accuracy_score(actual, prediction)),
        'f1_malignant': float(f1_score(actual, prediction, zero_division=0)),
        'recall_malignant': float(recall_score(actual, prediction, zero_division=0)),
        'roc_auc': float(roc_auc_score(actual, score)),
    }


def run(args):
    if args.rounds < 1 or args.epochs < 1 or args.batch < 1:
        raise ValueError('Rounds, epochs and batch size must be positive.')
    if args.mode == 'dp' and (args.noise <= 0 or args.clip <= 0):
        raise ValueError('DP mode requires positive noise and clipping.')
    if not 0 < args.delta < 1:
        raise ValueError('Delta must be between zero and one.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Choose a new output directory to preserve the previous run.')
    torch.set_num_threads(1)
    x, y, partitions, public, test, manifest = prepare_data(args.seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        initial = copy.deepcopy(model().state_dict())
    clients = []
    for indices in partitions:
        raw = model()
        raw.load_state_dict(initial)
        optimizer = torch.optim.Adam(raw.parameters(), lr=0.001)
        sampling = torch.Generator().manual_seed(secrets.randbits(63))
        loader = DataLoader(TensorDataset(x[indices], y[indices]),
                            batch_size=args.batch, shuffle=True, generator=sampling,
                            num_workers=0, drop_last=False)
        engine = None
        network = raw
        if args.mode == 'dp':
            engine = PrivacyEngine(accountant='rdp', secure_mode=False)
            network, optimizer, loader = engine.make_private(
                module=raw, optimizer=optimizer, data_loader=loader,
                noise_multiplier=args.noise, max_grad_norm=args.clip,
                poisson_sampling=True,
                noise_generator=torch.Generator().manual_seed(secrets.randbits(63)),
            )
        else:
            loader = DPDataLoader.from_data_loader(loader)
        clients.append({'raw': raw, 'network': network, 'optimizer': optimizer,
                        'loader': loader, 'engine': engine, 'steps': 0,
                        'weight': len(indices) / sum(map(len, partitions))})
    global_state = initial
    global_model = model()
    observations = []
    started = time.perf_counter()
    for round_index in range(1, args.rounds + 1):
        states = []
        epsilon = []
        for client in clients:
            client['raw'].load_state_dict(global_state)
            client['network'].train()
            previous_steps = client['steps']
            for _ in range(args.epochs):
                for inputs, labels in client['loader']:
                    client['optimizer'].zero_grad(set_to_none=True)
                    logits = client['network'](inputs)
                    # An empty Poisson batch still produces a counted noisy step.
                    loss = nn.functional.cross_entropy(logits, labels) if len(labels) else logits.sum()
                    loss.backward()
                    client['optimizer'].step()
                    client['steps'] += 1
            assert client['steps'] > previous_steps
            if client['engine'] is not None:
                accounted_steps = sum(n for _, _, n in client['engine'].accountant.history)
                assert accounted_steps == client['steps'], 'Privacy history was reset.'
                epsilon.append(float(client['engine'].get_epsilon(delta=args.delta)))
            states.append(copy.deepcopy(client['raw'].state_dict()))
        global_state = {key: sum(c['weight'] * state[key] for c, state in zip(clients, states))
                        for key in initial}
        global_model.load_state_dict(global_state)
        row = {'round': round_index, **public_metrics(global_model, x, y, public),
               'accounted_epsilon_opacus': max(epsilon) if epsilon else None}
        observations.append(row)
        print(json.dumps({'public_validation': row}), flush=True)
    elapsed = time.perf_counter() - started
    test_metrics = public_metrics(global_model, x, y, test)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'scikit-learn', 'scipy', 'opacus')}
    summary = {
        'implementation': 'standalone CPU feasibility pilot, not NEBULA end-to-end',
        'mode': args.mode, 'rounds': args.rounds, 'local_epochs': args.epochs,
        'model': '30-256-128-2 MLP',
        'model_parameters': sum(p.numel() for p in global_model.parameters()),
        'optimizer': 'Adam', 'learning_rate': 0.001,
        'batch_requested': args.batch, 'sample_rates': [c['loader'].sample_rate for c in clients],
        'public_normalization_factors': [getattr(c['optimizer'], 'expected_batch_size', None) for c in clients],
        'client_steps': [c['steps'] for c in clients],
        'training_wall_seconds': elapsed,
        'wall_clock_scope': 'federated loop including public validation, privacy queries and console output; excludes import/setup and final test',
        'local_optimizer_state': 'Adam moments persist between rounds in this standalone pilot',
        'cpu_threads': torch.get_num_threads(), 'device': 'cpu',
        'test_metrics': test_metrics, 'versions': versions,
        'python': platform.python_version(), 'system': platform.platform(),
        'noise_multiplier': args.noise if args.mode == 'dp' else None,
        'max_grad_norm': args.clip if args.mode == 'dp' else None,
        'delta': args.delta if args.mode == 'dp' else None,
        'accounted_epsilon_opacus': observations[-1]['accounted_epsilon_opacus'],
        'privacy_scope': ('no DP bound, non-private baseline' if args.mode == 'baseline' else
                          'nominal Opacus training accounting for one public-data pilot run; no clinical or production privacy certification'),
        'secure_mode': False,
        'limitations': ['single run, not the full three-repeat benchmark',
                        'synthetic institutions, not real multicenter data',
                        'no network, no NEBULA containers in this pilot',
                        'non-cryptographic noise generator in this feasibility environment'],
    }
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'public_validation.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)
    print(json.dumps({'completed': summary}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'dp'], required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=1.0)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
