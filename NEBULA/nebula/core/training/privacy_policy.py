"""Constant noise multipliers and public-reference adaptive clipping.

Fixed and adaptive modes share a pre-calibrated sigma and sampling plan.
Adaptive mode changes C from public feedback, with noise proportional to C.
Epsilon is a bound, not a privacy percentage.
"""
import math


def validate_dp_config(config):
    result = {
        "enabled": False, "mode": "fixed", "noise_multiplier": 1.0,
        "noise_end": 0.7, "max_grad_norm": 1.0, "target_epsilon": 8.0,
        "target_delta": 1e-5, "accountant": "rdp", "secure_mode": False,
        "poisson_sampling": True, "clipping": "flat",
        "adaptive_quantile": 0.5, "adaptive_smoothing": 0.5,
        "adaptive_min_clip": 0.1, "adaptive_max_clip": 5.0,
        "adaptive_max_records": 91,
    }
    result.update(config or {})
    for key in ("enabled", "secure_mode", "poisson_sampling"):
        if not isinstance(result[key], bool):
            raise ValueError(f"DP {key} must be a boolean")
    if result["mode"] not in {"fixed", "adaptive"}:
        raise ValueError("New DP runs require fixed or adaptive mode. Historical dynamic runs keep their original meaning.")
    for key in ("noise_multiplier", "noise_end", "max_grad_norm", "target_epsilon", "target_delta",
                "adaptive_quantile", "adaptive_smoothing", "adaptive_min_clip", "adaptive_max_clip"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"DP {key} must be finite and greater than zero")
    if not 0 < result["target_delta"] < 1:
        raise ValueError("DP delta must be between zero and one")
    if not 0 < result["adaptive_quantile"] < 1 or not 0 < result["adaptive_smoothing"] <= 1:
        raise ValueError("Adaptive quantile must be in (0, 1) and smoothing in (0, 1]")
    if result["adaptive_min_clip"] > result["adaptive_max_clip"]:
        raise ValueError("Adaptive minimum C must not exceed maximum C")
    if result["mode"] == "adaptive" and not result["adaptive_min_clip"] <= result["max_grad_norm"] <= result["adaptive_max_clip"]:
        raise ValueError("Initial C must be inside the adaptive clipping limits")
    records = result["adaptive_max_records"]
    if isinstance(records, bool) or not isinstance(records, int) or not 1 <= records <= 91:
        raise ValueError("Adaptive public reference size must be an integer from 1 to 91")
    if result["accountant"] != "rdp" or not result["poisson_sampling"] or result["clipping"] != "flat":
        raise ValueError("Project 2 requires RDP, Poisson sampling and flat clipping")
    return result


def noise_schedule(config, rounds):
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("At least one round is required")
    start = config["noise_multiplier"]
    return [start] * rounds


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
    """Calibrate a constant sigma to the same epsilon ceiling in both modes.

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
