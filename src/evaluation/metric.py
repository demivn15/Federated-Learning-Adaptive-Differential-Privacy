import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

def compute_mse(original_img: torch.Tensor, reconstructed_img: torch.Tensor) -> float:
    """
    Computes Mean Squared Error (MSE) between original raw image patch and reconstructed output.
    Target: Higher MSE under Adaptive DP indicates successful disruption of reconstruction.
    """
    if original_img.shape != reconstructed_img.shape:
        raise ValueError(f"Shape mismatch: {original_img.shape} vs {reconstructed_img.shape}")
    
    return float(F.mse_loss(original_img, reconstructed_img, reduction='mean').item())


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Computes Structural Similarity Index (SSIM) on a scale of [-1, 1].
    Target: SSIM < 0.20 confirms zero recognizable cellular morphology or tumor features.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
        
    img1, img2 = img1.float(), img2.float()
    channel = img1.size(1)
    
    def gaussian(w_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - w_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(w_size)]))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def compute_mia_metrics(y_true_membership: np.ndarray, y_pred_scores: np.ndarray):
    """
    Computes Membership Inference Attack metrics (MIA-AUC, MIA Accuracy).
    Target: MIA-AUC near 0.50 (random guessing) proves strong indistinguishability.
    """
    mia_auc = float(roc_auc_score(y_true_membership, y_pred_scores))
    binary_preds = (y_pred_scores > 0.5).astype(int) if np.max(y_pred_scores) <= 1.0 else (y_pred_scores > 0.0).astype(int)
    mia_acc = float(accuracy_score(y_true_membership, binary_preds))
    
    return {"mia_auc": mia_auc, "mia_acc": mia_acc}


def compute_utility_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray):
    """
    Computes diagnostic performance metrics (AUROC >= 0.88, Sensitivity >= 90%).
    """
    auroc = float(roc_auc_score(y_true, y_pred_probs))
    binary_preds = (y_pred_probs >= 0.5).astype(int)
    _, recall, f1, _ = precision_recall_fscore_support(y_true, binary_preds, average='binary')
    
    return {"auroc": auroc, "sensitivity": float(recall), "f1_score": float(f1)}
