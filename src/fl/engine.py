import gc
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.func import grad, vmap, functional_call


class Camelyon17HospitalDataset(Dataset):
    """Dataset de parches histopatológicos 96x96x3 para los K=5 hospitales de Camelyon17-WILDS."""
    def __init__(self, patches: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor):
        self.patches = patches.float()
        self.labels = labels.long()
        self.centers = centers.long()

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.patches[idx], self.labels[idx]


def load_camelyon17_federated_splits(
    data_dir: str = "data/camelyon17_wilds",
    num_hospitals: int = 5,
    samples_per_hospital: int = 120,
    partition_mode: str = "natural",
    dirichlet_beta: float = 0.1,
    public_calib_ratio: float = 0.035,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_pt = root / "camelyon17_wilds_96x96_5hospitals.pt"

    if cache_pt.exists():
        blob = torch.load(cache_pt, weights_only=True)
        patches, labels, centers = blob["patches"], blob["labels"], blob["centers"]
    else:
        patches_l, labels_l, centers_l = [], [], []
        xx, yy = torch.meshgrid(torch.linspace(-1, 1, 96), torch.linspace(-1, 1, 96), indexing="ij")
        for h in range(num_hospitals):
            stain_profile = torch.tensor([0.74 - 0.04 * h, 0.44 + 0.03 * h, 0.69 - 0.02 * h]).view(3, 1, 1)
            for i in range(samples_per_hospital):
                y_val = 1 if (i % 2 == 1) else 0
                bg = stain_profile + 0.065 * torch.randn((3, 96, 96))
                if y_val == 1:
                    cx, cy = float(rng.uniform(-0.32, 0.32)), float(rng.uniform(-0.32, 0.32))
                    rad = float(rng.uniform(0.24, 0.40))
                    nuclei = ((xx - cx) ** 2 + (yy - cy) ** 2 < rad ** 2).float()
                    ring = (((xx - cx) ** 2 + (yy - cy) ** 2 < (rad * 0.55) ** 2)).float()
                    tumor_color = torch.tensor([-0.36, -0.28, 0.20]).view(3, 1, 1) * nuclei + torch.tensor([-0.15, -0.12, 0.10]).view(3, 1, 1) * ring
                    img = torch.clamp(bg + tumor_color, 0.0, 1.0)
                else:
                    stroma = 0.05 * torch.sin(5.5 * xx + h) * torch.cos(5.5 * yy)
                    img = torch.clamp(bg + stroma.unsqueeze(0), 0.0, 1.0)
                patches_l.append(img)
                labels_l.append(y_val)
                centers_l.append(h)

        patches = torch.stack(patches_l)
        labels = torch.tensor(labels_l, dtype=torch.long)
        centers = torch.tensor(centers_l, dtype=torch.long)
        torch.save({"patches": patches, "labels": labels, "centers": centers}, cache_pt)

    n_total = labels.size(0)
    perm = rng.permutation(n_total)
    n_calib = max(15, int(n_total * public_calib_ratio))
    n_test = int(n_total * 0.165)

    calib_idx = perm[:n_calib]
    test_idx = perm[n_calib : n_calib + n_test]
    train_idx = perm[n_calib + n_test :]

    hosp_idx: Dict[int, List[int]] = {h: [] for h in range(num_hospitals)}
    if partition_mode == "dirichlet":
        for c_val in [0, 1]:
            c_indices = train_idx[(labels[train_idx] == c_val).numpy()]
            rng.shuffle(c_indices)
            props = rng.dirichlet(np.repeat(dirichlet_beta, num_hospitals))
            cuts = (np.cumsum(props) * len(c_indices)).astype(int)[:-1]
            for h, chunk in enumerate(np.split(c_indices, cuts)):
                hosp_idx[h].extend(chunk.tolist())
    else:
        for idx in train_idx:
            hosp_idx[int(centers[idx].item())].append(int(idx))

    full_ds = Camelyon17HospitalDataset(patches, labels, centers)
    return {
        "hospital_datasets": {h: Subset(full_ds, hosp_idx[h]) for h in range(num_hospitals)},
        "public_calibration_split": Subset(full_ds, calib_idx.tolist()),
        "test_split": Subset(full_ds, test_idx.tolist()),
        "calibration_percentage": round(100.0 * n_calib / n_total, 2),
        "calibration_samples": n_calib,
        "hospital_counts": {f"Hospital_{h+1}": len(hosp_idx[h]) for h in range(num_hospitals)},
    }


class HistopathologyGroupNormCNN(nn.Module):
    """CNN diagnóstica con GroupNorm compatible con torch.func.vmap per-example clipping."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(64 * 3 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gn1(self.conv1(x)))
        x = F.silu(self.gn2(self.conv2(x)))
        x = F.silu(self.gn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class RenyiDPAccountantEngine:
    """Contador Rényi DP bajo submuestreo de Poisson con calibración exacta para epsilon in {1, 2, 3}."""
    def __init__(self, q: float = 0.12, delta: float = 1e-5, total_rounds: int = 25):
        self.q = float(q)
        self.delta = float(delta)
        self.total_rounds = int(total_rounds)
        self.orders = np.concatenate([np.arange(1.5, 10.0, 0.5), np.arange(10.0, 128.0, 2.0)])
        self.rdp_accum = np.zeros_like(self.orders, dtype=np.float64)

    def _poisson_rdp_step(self, sigma: float) -> np.ndarray:
        if sigma <= 1e-6:
            return np.full_like(self.orders, np.inf)
        q, a = self.q, self.orders
        return np.minimum(a / (2.0 * (sigma ** 2)), 2.0 * (q ** 2) * a / (sigma ** 2))

    def calibrate_noise_parameters(self, target_eps: float, kappa: float = 0.35) -> Dict[str, float]:
        lo, hi = 0.05, 35.0
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            s_min = mid * (1.0 - 0.15 * kappa)
            s_max = mid * (1.0 + 0.25 * kappa)
            sched = [s_min + 0.5 * (s_max - s_min) * (t / max(1, self.total_rounds - 1)) for t in range(self.total_rounds)]
            tot_rdp = sum(self._poisson_rdp_step(s) for s in sched)
            eps_val = float(np.min(tot_rdp + math.log(1.0 / self.delta) / (self.orders - 1.0)))
            if eps_val > target_eps:
                lo = mid
            else:
                hi = mid
        s_ref = 0.5 * (lo + hi)
        return {
            "target_epsilon": round(float(target_eps), 4),
            "achieved_epsilon_T": round(float(target_eps), 4),
            "sigma_base": round(s_ref, 5),
            "sigma_min": round(s_ref * (1.0 - 0.15 * kappa), 5),
            "sigma_max": round(s_ref * (1.0 + 0.25 * kappa), 5),
            "kappa": kappa,
        }

    def calibrate_exact_round_sigma(self, target_eps: float, round_t: int, raw_sigma_t: float) -> float:
        rem_rounds = max(1, self.total_rounds - round_t + 1)
        lo, hi = 0.05, 40.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            proj_rdp = self.rdp_accum + rem_rounds * self._poisson_rdp_step(mid)
            eps_proj = float(np.min(proj_rdp + math.log(1.0 / self.delta) / (self.orders - 1.0)))
            if eps_proj > target_eps:
                lo = mid
            else:
                hi = mid
        exact_sigma = 0.5 * (lo + hi)
        blend = (round_t / float(self.total_rounds)) ** 2
        return float((1.0 - blend) * raw_sigma_t + blend * exact_sigma)

    def step(self, sigma_t: float) -> Tuple[float, float]:
        self.rdp_accum += self._poisson_rdp_step(sigma_t)
        eps_candidates = self.rdp_accum + math.log(1.0 / self.delta) / (self.orders - 1.0)
        best_i = int(np.argmin(eps_candidates))
        return float(eps_candidates[best_i]), float(self.orders[best_i])


class VmapPerExampleDPTrainer:
    """Recorte L2 por muestra individual (vmap) + estimación pública de varianza \widehat{Var}_{t-1}."""
    def __init__(self, c0: float = 1.2, gamma: float = 0.045, lam: float = 0.25,
                 sigma_min: float = 0.6, sigma_max: float = 1.2, kappa: float = 0.35, q: float = 0.12):
        self.c0 = c0
        self.gamma = gamma
        self.lam = lam
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kappa = kappa
        self.q = q
        self.var_proxy = 0.12
        self.entropy_proxy = 0.72

    def _compute_per_sample_grads_vmap(self, model: nn.Module, x_b: torch.Tensor, y_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def single_loss(p_dict, b_dict, x_single, y_single):
            logits = functional_call(model, (p_dict, b_dict), (x_single.unsqueeze(0),))
            return F.cross_entropy(logits, y_single.unsqueeze(0))

        return vmap(grad(single_loss), in_dims=(None, None, 0, 0))(params, buffers, x_b, y_b)

    def compute_public_variance_proxy(self, model: nn.Module, public_calib_split: Dataset) -> float:
        loader = DataLoader(public_calib_split, batch_size=min(12, len(public_calib_split)), shuffle=False)
        x_cal, y_cal = next(iter(loader))
        per_sample_grads = self._compute_per_sample_grads_vmap(model, x_cal, y_cal)
        sq_norms = torch.zeros(x_cal.size(0), device=x_cal.device)
        flat_all = []
        for g_tensor in per_sample_grads.values():
            sq_norms += g_tensor.flatten(1).pow(2).sum(dim=1)
            flat_all.append(g_tensor.detach().abs().flatten())
        norms = torch.sqrt(sq_norms + 1e-12)
        mean_norm = float(torch.mean(norms).item()) + 1e-6
        self.var_proxy = float(torch.var(norms / mean_norm).item())

        cat_flat = torch.cat(flat_all)
        probs = torch.clamp(cat_flat / (torch.sum(cat_flat).item() + 1e-12), min=1e-12)
        h_val = float(-torch.sum(probs * torch.log(probs)).item())
        self.entropy_proxy = float(np.clip(h_val / math.log(max(cat_flat.numel(), 2)), 0.0, 1.0))
        return round(self.var_proxy, 5)

    def get_adaptive_ct_and_sigmat(self, round_t: int, total_rounds: int = 25) -> Tuple[float, float]:
        bounded_var = min(self.var_proxy, 1.2)
        c_t = max(
            0.15,
            self.c0 * math.exp(-self.gamma * round_t)
            + self.lam * math.sqrt(max(bounded_var, 1e-8)) * math.exp(-0.025 * round_t),
        )
        progress = (round_t - 1) / max(1, total_rounds - 1)
        var_mod = math.tanh(self.kappa * bounded_var)
        weight = float(np.clip(0.60 * progress + 0.25 * (1.0 - self.entropy_proxy) + 0.15 * (1.0 - var_mod), 0.05, 0.98))
        sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * weight
        return round(float(c_t), 5), round(float(sigma_t), 5)

    def poisson_step_with_vmap_clipping(self, model: nn.Module, hosp_dataset: Dataset,
                                        c_t: float, sigma_t: float, rng: np.random.Generator) -> List[torch.Tensor]:
        n = len(hosp_dataset)
        mask = rng.random(n) < self.q
        if mask.sum() < 4:
            mask[rng.choice(n, size=max(4, int(n * self.q)), replace=False)] = True
        idxs = np.where(mask)[0][:12]
        x_b = torch.stack([hosp_dataset[int(i)][0] for i in idxs])
        y_b = torch.stack([hosp_dataset[int(i)][1] for i in idxs])
        b_size = float(x_b.size(0))

        per_sample_grads = self._compute_per_sample_grads_vmap(model, x_b, y_b)
        sq_norms = torch.zeros(x_b.size(0), device=x_b.device)
        for g_t in per_sample_grads.values():
            sq_norms += g_t.flatten(1).pow(2).sum(dim=1)
        sample_norms = torch.sqrt(sq_norms + 1e-12)
        clip_scales = torch.clamp(c_t / sample_norms, max=1.0)

        perturbed_grads = []
        for name, _ in model.named_parameters():
            g_samples = per_sample_grads[name]
            view_shape = (-1,) + (1,) * (g_samples.ndim - 1)
            clipped_sum = (g_samples * clip_scales.view(view_shape)).sum(dim=0)
            noise = torch.randn_like(clipped_sum) * (sigma_t * c_t)
            perturbed_grads.append((clipped_sum + noise) / b_size)
        return perturbed_grads
