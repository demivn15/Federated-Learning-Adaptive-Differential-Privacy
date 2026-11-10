"""Public round schedules and cumulative RDP accounting for Project 2.

The schedule depends on round numbers and accountant state, never on private
loss, accuracy or gradients. Epsilon is a bound, not a privacy percentage.
"""
import math


def validate_dp_config(config):
    result = {
        "enabled": False, "mode": "fixed", "noise_multiplier": 1.0,
        "noise_end": 0.7, "max_grad_norm": 1.0, "target_epsilon": 8.0,
        "target_delta": 1e-5, "accountant": "rdp", "secure_mode": False,
        "poisson_sampling": True, "clipping": "flat",
    }
    result.update(config or {})
    for key in ("enabled", "secure_mode", "poisson_sampling"):
        if not isinstance(result[key], bool):
            raise ValueError(f"DP {key} must be a boolean")
    if result["mode"] not in {"fixed", "dynamic"}:
        raise ValueError("DP mode must be fixed or dynamic")
    for key in ("noise_multiplier", "noise_end", "max_grad_norm", "target_epsilon", "target_delta"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"DP {key} must be finite and greater than zero")
    if not 0 < result["target_delta"] < 1:
        raise ValueError("DP delta must be between zero and one")
    if result["accountant"] != "rdp" or not result["poisson_sampling"] or result["clipping"] != "flat":
        raise ValueError("Project 2 requires RDP, Poisson sampling and flat clipping")
    return result


def noise_schedule(config, rounds):
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("At least one round is required")
    start = config["noise_multiplier"]
    end = config["noise_end"] if config["mode"] == "dynamic" else start
    return [start + (end - start) * r / max(1, rounds - 1) for r in range(rounds)]


def projected_epsilon(history, schedule, sample_rate, steps_per_round, delta):
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    accountant.history = list(history) + [
        (float(sigma), float(sample_rate), int(steps_per_round)) for sigma in schedule
    ]
    # Integer orders are a subset of Opacus' default orders. This gives a
    # conservative calibration bound and avoids slow fractional-order searches
    # on the student's CPU. Final reporting still uses Opacus' default grid.
    return accountant.get_epsilon(delta=delta, alphas=[2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 63])


def calibrate_schedule(config, rounds, sample_rate, steps_per_round):
    """Scale the complete fixed/public schedule to the same epsilon ceiling.

    Calibration occurs before the first private update. The small margin avoids
    crossing the ceiling due to floating-point conversion of the RDP bound.
    """
    schedule = noise_schedule(config, rounds)
    target = config["target_epsilon"] * 0.995
    def epsilon(scale):
        return projected_epsilon([], [s * scale for s in schedule], sample_rate,
                                 steps_per_round, config["target_delta"])
    low, high = 0.001, 1.0
    while epsilon(high) > target:
        high *= 2
        if high > 16384:
            raise ValueError("Could not calibrate this privacy budget")
    for _ in range(16):
        mid = (low + high) / 2
        if epsilon(mid) > target:
            low = mid
        else:
            high = mid
    calibrated = [s * high for s in schedule]
    return calibrated
