"""Run with .venv-pilot/Scripts/python.exe -B verify_privacy.py from this folder."""
import copy
import importlib.util
import json
import os
import secrets
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("PROJECT2_REPO", str(HERE / "nebula")))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus.data_loader import DPDataLoader
from nebula.core.training.dp import DifferentialPrivacyPlugin, SimpleDPState
from nebula.core.training.privacy_policy import noise_schedule, validate_dp_config
from nebula.core.datasets.breast_cancer.research_data import prepare_arrays
from nebula.core.training.research_results import evaluate_public
from nebula.core.training.adaptive_clipping import PublicAdaptiveClipper


class PrivacyTests(unittest.TestCase):
    def test_invalid_parameters_and_constant_sigma(self):
        for bad in ({"noise_multiplier": 0}, {"target_epsilon": float("nan")},
                    {"target_delta": 1}, {"mode": "accuracy_feedback"},
                    {"enabled": "false"}, {"accountant": "gdp"}, {"mode": "dynamic"},
                    {"adaptive_quantile": 1}, {"adaptive_smoothing": 0},
                    {"adaptive_min_clip": 6}, {"adaptive_max_records": True},
                    {"mode": "adaptive", "max_grad_norm": 10}):
            with self.assertRaises(ValueError):
                validate_dp_config(bad)
        self.assertEqual(noise_schedule(validate_dp_config({}), 3), [1, 1, 1])
        values = noise_schedule(validate_dp_config({"mode": "adaptive"}), 3)
        self.assertEqual(values, [1, 1, 1])

    def test_adaptation_responds_to_public_model_and_preserves_parameters(self):
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        model.train()
        model.weight.grad = torch.ones_like(model.weight)
        original_gradient = model.weight.grad.clone()
        settings = validate_dp_config({"mode": "adaptive", "adaptive_smoothing": 1})
        controller = PublicAdaptiveClipper(settings)
        public = TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))
        initial = copy.deepcopy(model.state_dict())
        clip1, record1 = controller.select(model, public, designated_public=True)
        self.assertAlmostEqual(clip1, 2 ** -.5, places=6)
        self.assertEqual(record1["reference_records"], 4)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.weight.grad, original_gradient))
        for key, value in initial.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        with torch.no_grad():
            model.bias[0] = 8
        clip2, record2 = controller.select(model, public, designated_public=True)
        self.assertLess(clip2, clip1)
        self.assertEqual(clip2, settings["adaptive_min_clip"])
        self.assertEqual(record2["previous_clip"], clip1)
        for dataset, flag in ((public, False), (None, True)):
            with self.assertRaises(ValueError):
                controller.select(model, dataset, designated_public=flag)

    def test_adaptation_requires_public_reference(self):
        model = torch.nn.Linear(3, 2)
        data = TensorDataset(torch.randn(16, 3), torch.zeros(16, dtype=torch.long))
        state = SimpleDPState()
        state.extras.update(dataloader=DataLoader(data, batch_size=8), epochs=1, round_index=0)
        plugin = DifferentialPrivacyPlugin(mode="adaptive", total_rounds=1)
        with self.assertRaisesRegex(ValueError, "public reference"):
            plugin.on_train_start(model, torch.optim.SGD(model.parameters(), lr=.01), state)
        self.assertEqual(plugin._privacy_engine.accountant.history, [])

    def test_public_split_and_non_iid(self):
        x, y, parts, public, test, manifest = prepare_arrays()
        self.assertEqual(x.shape, (569, 30))
        self.assertEqual([sum(map(len, parts)), len(public), len(test)], [364, 91, 114])
        self.assertEqual(len(set(i for p in parts for i in p)), 364)
        self.assertFalse(set(public) & set(test))
        self.assertEqual(manifest["label_mapping"]["0"], "malignant")
        proportions = [np.mean(y[p] == 0) for p in parts]
        self.assertGreater(max(proportions) - min(proportions), 0.1)
        again = prepare_arrays()
        self.assertEqual(parts, again[2])
        self.assertTrue(np.allclose(x[public].mean(axis=0), 0, atol=1e-5))
        self.assertEqual(sum(map(len, prepare_arrays(iid=True)[2])), 364)

    def test_actual_opacus_rounds_budget_and_step_guard(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        data = TensorDataset(torch.randn(32, 3), torch.randint(0, 2, (32,)))
        final_epsilons = []
        for mode in ("fixed", "adaptive"):
            model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
            plugin = DifferentialPrivacyPlugin(mode=mode, target_epsilon=5, total_rounds=2, noise_end=0.5)
            epsilons = []
            for round_index in range(2):
                state = SimpleDPState()
                state.extras.update(dataloader=DPDataLoader.from_data_loader(DataLoader(data, batch_size=8, shuffle=True)),
                                    epochs=1, round_index=round_index,
                                    public_reference_dataset=TensorDataset(torch.ones(8, 3), torch.zeros(8, dtype=torch.long)),
                                    public_reference_is_public=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                plugin.on_train_start(model, optimizer, state)
                private = state.extras["model"]
                optimizer = state.extras["optimizer"]
                for inputs, labels in state.extras["dataloader"]:
                    optimizer.zero_grad()
                    logits = private(inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels) if len(labels) else logits.sum()
                    loss.backward()
                    optimizer.step()
                before = copy.deepcopy(plugin._privacy_engine.accountant.history)
                optimizer.zero_grad()
                private(data.tensors[0][:2]).sum().backward()
                with self.assertRaises(ValueError):
                    optimizer.step()
                self.assertEqual(plugin._privacy_engine.accountant.history, before)
                plugin.on_train_end(state)
                epsilons.append(state.extras["dp_epsilon"])
                self.assertLessEqual(epsilons[-1], 5)
                self.assertEqual(sum(n for _, _, n in before), (round_index + 1) * 4)
                self.assertEqual(optimizer.max_grad_norm, state.extras["clip"])
                if mode == "adaptive":
                    self.assertEqual(state.extras["adaptation"]["reference_records"], 8)
            self.assertGreater(epsilons[1], epsilons[0])
            self.assertEqual(plugin.schedule[0], plugin.schedule[1])
            final_epsilons.append(epsilons[-1])
            with self.assertRaises(ValueError):
                plugin.on_train_start(model, torch.optim.Adam(model.parameters()), state)
        self.assertEqual(final_epsilons[0], final_epsilons[1])

    def test_metrics_treat_malignant_as_positive(self):
        class KnownModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, x):
                return x * self.weight
        model = KnownModel()
        dataset = TensorDataset(torch.tensor([[3., 0.], [0., 3.], [0., 3.], [0., 3.]]),
                                torch.tensor([0, 0, 1, 1]))
        result = evaluate_public(model, dataset)
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["malignant_recall"], .5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertTrue(model.training)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "HDF5 integration requires the installed Ubuntu environment")
    def test_hdf5_partitions_for_three_trainers_and_public_server(self):
        import h5py
        from types import SimpleNamespace
        from nebula.core.datasets.breast_cancer.research_data import write_research_partitions
        root = HERE / "test_artifacts" / ("hdf5_" + secrets.token_hex(4))
        root.mkdir(parents=True)
        for i in range(4):
            (root / f"participant_{i}.json").write_text(json.dumps({"device_args": {
                "idx": i, "role": "server" if i == 0 else "trainer"}}), encoding="utf-8")
        ds = SimpleNamespace(config_dir=str(root), iid=False, partition="dirichlet", partition_parameter=.5, seed=42)
        manifest = write_research_partitions(ds)
        total = 0
        for i in range(1, 4):
            with h5py.File(root / f"participant_{i}_train.h5", "r") as f:
                total += len(f["train_targets"])
                self.assertEqual(f["train_data"].shape[1], 30)
                self.assertEqual(f["train_data"].attrs["num_classes"], 2)
        self.assertEqual(total, 364)
        with h5py.File(root / "participant_0_train.h5", "r") as f:
            self.assertEqual(len(f["train_targets"]), 91)
        with h5py.File(root / "global_test.h5", "r") as f:
            self.assertEqual(len(f["test_targets"]), 114)
        self.assertEqual(manifest["protocol"], "project2-public-evaluation-v3")

    @unittest.skipUnless(importlib.util.find_spec("lightning"), "Lightning integration requires the installed Ubuntu environment")
    def test_actual_nebula_lightning_private_training(self):
        from datetime import datetime
        from types import SimpleNamespace
        from nebula.core.datasets.datamodule import DataModule
        from nebula.core.models.breast_cancer.mlp import BreastCancerModelMLP
        from nebula.core.training.lightning_dp import LightningDP
        name = "lightning_check_" + secrets.token_hex(4)
        root = HERE / "test_artifacts"
        config = SimpleNamespace(participant={
            "device_args": {"idx": 1, "role": "trainer", "accelerator": "cpu", "gpu_id": []},
            "scenario_args": {"name": name, "rounds": 2, "random_seed": 42,
                              "start_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S")},
            "tracking_args": {"log_dir": str(root), "local_tracking": "basic"},
            "training_args": {"dp": validate_dp_config({"enabled": True, "mode": "adaptive"})}})
        data = TensorDataset(torch.randn(32, 30), torch.tensor([0, 1] * 16))
        public = TensorDataset(torch.randn(16, 30), torch.tensor([0, 1] * 8))
        dm = DataModule(data, list(range(32)), public, list(range(16)), public, list(range(16)), {}, batch_size=8)
        dm.public_validation = public
        model = BreastCancerModelMLP()
        model.research_public_evaluation = True
        trainer = LightningDP(model, dm, config)
        trainer.set_epochs(1)
        for r in range(2):
            trainer.set_current_round(r)
            trainer.create_trainer()
            loss, accuracy, train_accuracy = trainer._train_sync()
            self.assertIsNone(loss)
            self.assertIsNone(train_accuracy)
            self.assertEqual(model.get_latest_train_metrics(), {})
            self.assertIsNotNone(accuracy)
            self.assertLessEqual(trainer.dp_epsilon, 8)
        self.assertEqual(trainer._dp_plugin.completed_rounds, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
