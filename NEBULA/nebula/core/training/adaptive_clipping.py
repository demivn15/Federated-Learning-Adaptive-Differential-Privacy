"""Round-wise clipping feedback from designated public reference records.

Only the public reference set and the incoming, already-private global model
enter this controller. Private training gradients and the final test do not.
Keep sigma and the sampling plan fixed. Noise on the clipped sum has standard
deviation sigma * C, so the RDP increment does not depend on the selected C.
This is public-reference adaptation, not privately estimated local quantiles.
"""
import math


class PublicAdaptiveClipper:
    def __init__(self, settings):
        self.settings = settings
        self.current = settings["max_grad_norm"]

    def select(self, model, dataset, *, designated_public=False):
        import torch
        from torch.nn.functional import cross_entropy

        if designated_public is not True or dataset is None:
            raise ValueError("Adaptive DP requires a designated public reference dataset, never private validation or final test")
        count = min(len(dataset), self.settings["adaptive_max_records"])
        if count < 1:
            raise ValueError("The public adaptation reference is empty")
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Adaptive clipping requires trainable model parameters")
        device, dtype = parameters[0].device, parameters[0].dtype
        previous_training = model.training
        norms = []
        try:
            # Deterministic indexed reads avoid changing the training sampler/RNG.
            model.eval()
            with torch.enable_grad():
                for index in range(count):
                    inputs, target = dataset[index]
                    inputs = torch.as_tensor(inputs, device=device, dtype=dtype).unsqueeze(0)
                    target = torch.as_tensor(target, device=device, dtype=torch.long).reshape(1)
                    loss = cross_entropy(model(inputs), target)
                    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
                    squared_norm = sum(g.detach().double().square().sum() for g in gradients)
                    norm = float(squared_norm.sqrt().item())
                    if not math.isfinite(norm):
                        raise ValueError("Non-finite gradient norm on the public adaptation reference")
                    norms.append(norm)
        finally:
            model.train(previous_training)

        quantile = float(torch.quantile(torch.tensor(norms, dtype=torch.float64),
                                       self.settings["adaptive_quantile"]).item())
        lower, upper = self.settings["adaptive_min_clip"], self.settings["adaptive_max_clip"]
        target = min(upper, max(lower, quantile))
        weight = self.settings["adaptive_smoothing"]
        previous = self.current
        self.current = min(upper, max(lower, (1 - weight) * previous + weight * target))
        return self.current, {
            "policy": "public-reference-gradient-quantile-v1",
            "source": "designated public calibration records and incoming DP model",
            "reference_records": count, "quantile": self.settings["adaptive_quantile"],
            "observed_quantile": quantile, "target_clip": target,
            "previous_clip": previous, "selected_clip": self.current,
            "smoothing": weight, "min_clip": lower, "max_clip": upper,
        }
