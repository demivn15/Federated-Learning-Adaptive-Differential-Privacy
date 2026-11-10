"""Opacus integration with a public noise schedule and cumulative accountant."""
import math
import secrets

from nebula.core.training.privacy_policy import calibrate_schedule, projected_epsilon, validate_dp_config
from nebula.core.training.research_results import append_record


class SimpleDPState:
    def __init__(self):
        self.extras = {}


class DifferentialPrivacyPlugin:
    name = "differential_privacy"

    def __init__(self, *, noise_multiplier=1.0, max_grad_norm=1.0, target_delta=1e-5,
                 accountant="rdp", secure_mode=False, poisson_sampling=True, clipping="flat",
                 mode="fixed", noise_end=0.7, target_epsilon=8.0, total_rounds=3, config=None):
        self.settings = validate_dp_config(dict(enabled=True, noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm, target_delta=target_delta, accountant=accountant,
            secure_mode=secure_mode, poisson_sampling=poisson_sampling, clipping=clipping,
            mode=mode, noise_end=noise_end, target_epsilon=target_epsilon))
        self.total_rounds = int(total_rounds)
        self.config = config
        self._privacy_engine = None
        self._noise_generator = None
        self._sample_generator = None
        self.schedule = None
        self.completed_rounds = 0
        self._shape = None

    def on_train_start(self, model, optimizer, state):
        import torch
        from opacus import PrivacyEngine
        loader = state.extras["dataloader"]
        epochs = int(state.extras.get("epochs", 1))
        round_index = int(state.extras.get("round_index", self.completed_rounds))
        if round_index != self.completed_rounds or round_index >= self.total_rounds:
            raise ValueError("DP rounds must be consecutive, with one persistent accountant")
        if len(loader) < 1 or epochs < 1:
            raise ValueError("DP needs at least one batch and one epoch")
        q = 1 / len(loader)
        steps = len(loader) * epochs
        if self._privacy_engine is None:
            if self.config is not None:
                from pathlib import Path
                p = self.config.participant
                journal = Path(p["tracking_args"]["log_dir"]) / p["scenario_args"]["name"] / f"participant_{p['device_args']['idx']}_privacy.jsonl"
                if journal.exists() and journal.stat().st_size:
                    raise ValueError("This scenario has a privacy history. Start a NEW scenario instead of resetting its accountant.")
            self._privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=self.settings["secure_mode"])
            if not self.settings["secure_mode"]:
                # Independent seeds are not exported. This experimental mode is not CSPRNG.
                self._noise_generator = torch.Generator(device=next(model.parameters()).device).manual_seed(secrets.randbits(63))
                self._sample_generator = torch.Generator().manual_seed(secrets.randbits(63))
            self.schedule = calibrate_schedule(self.settings, self.total_rounds, q, steps)
            self._shape = (q, steps)
        if self._shape != (q, steps):
            raise ValueError("Public sampling plan changed. Create a new calibrated scenario.")
        engine = self._privacy_engine
        sigma = self.schedule[round_index]
        forecast = projected_epsilon(engine.accountant.history, self.schedule[round_index:],
                                     q, steps, self.settings["target_delta"])
        if not math.isfinite(forecast) or forecast > self.settings["target_epsilon"]:
            raise ValueError("Planned training exceeds the cumulative privacy budget")
        model.train()
        if self._sample_generator is not None:
            loader.generator = self._sample_generator
        private_model, private_optimizer, private_loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=sigma, max_grad_norm=self.settings["max_grad_norm"],
            poisson_sampling=True, clipping="flat", noise_generator=self._noise_generator)
        expected = sum(n for _, _, n in engine.accountant.history) + steps
        account_hook = engine.accountant.get_optimizer_hook_fn(sample_rate=q)
        def bounded_accounting(dp_optimizer):
            count = sum(n for _, _, n in engine.accountant.history)
            if dp_optimizer.accumulated_iterations != 1 or count >= expected:
                raise ValueError("An unplanned optimizer step would invalidate the privacy plan")
            account_hook(dp_optimizer)
        private_optimizer.attach_step_hook(bounded_accounting)
        state.extras.update(privacy_engine=engine, model=private_model, optimizer=private_optimizer,
                            dataloader=private_loader, expected_steps=expected, sigma=sigma,
                            sample_rate=q, round_index=round_index, steps_this_round=steps)

    def on_train_end(self, state):
        engine = state.extras.get("privacy_engine")
        private_model = state.extras.get("model")
        try:
            if engine is not None:
                count = sum(n for _, _, n in engine.accountant.history)
                epsilon = float(engine.get_epsilon(delta=self.settings["target_delta"]))
                complete = count == state.extras["expected_steps"]
                state.extras.update(dp_epsilon=epsilon, dp_delta=self.settings["target_delta"])
                if self.config is not None:
                    append_record(self.config, "privacy", {
                        "round": state.extras["round_index"] + 1, "complete": complete,
                        "mode": self.settings["mode"], "epsilon": epsilon,
                        "epsilon_limit": self.settings["target_epsilon"], "delta": self.settings["target_delta"],
                        "sigma": state.extras["sigma"], "clip": self.settings["max_grad_norm"],
                        "sample_rate": state.extras["sample_rate"], "cumulative_steps": count,
                        "secure_rng": self.settings["secure_mode"], "noise_schedule": self.schedule,
                        "scope": "Conditional record-level training accountant, fixed public allocation. No hospital-level guarantee."})
                if not complete or epsilon > self.settings["target_epsilon"]:
                    raise ValueError("Incomplete DP round or exceeded privacy budget. Do not release this run as complete.")
                self.completed_rounds += 1
        finally:
            if private_model is not None:
                private_model.zero_grad(set_to_none=True)
                private_model.forbid_grad_accumulation()
                private_model.remove_hooks()
