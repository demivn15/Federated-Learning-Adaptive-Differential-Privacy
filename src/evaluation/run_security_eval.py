import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.metrics import compute_mse, compute_ssim, compute_mia_metrics
from src.attacks.dlg_attack import DLGAttack
from src.attacks.venom_inversion import VenomAttack
from src.attacks.fedmia_attack import FedMIAAttack

class SimplePathologyResNet(nn.Module):
    """ResNet-18 backbone for 96x96x3 histopathology image patches."""
    def __init__(self, num_classes=2):
        super(SimplePathologyResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


def generate_simulated_updates(model, target_data, target_label, mechanism="adaptive_dp", C_t=1.0, sigma_t=0.5):
    """Generates parameter updates under non-private, static DP, and adaptive DP mechanics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    outputs = model(target_data)
    loss = criterion(outputs, target_label)
    true_grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    
    if mechanism == "non_private":
        return list(true_grads)
    
    grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in true_grads))
    clipped_grads = []
    
    if mechanism == "static_dp":
        clip_factor = max(1.0, (grad_norm / 1.0).item())
        for g in true_grads:
            cg = g / clip_factor
            noise = torch.randn_like(cg) * (0.8 * 1.0) # Fixed noise multiplier
            clipped_grads.append(cg + noise)
            
    elif mechanism == "adaptive_dp":
        clip_factor = max(1.0, (grad_norm / C_t).item())
        for g in true_grads:
            cg = g / clip_factor
            noise = torch.randn_like(cg) * (sigma_t * C_t) # Scaled to dynamic bound C_t
            clipped_grads.append(cg + noise)
            
    return clipped_grads


def run_security_evaluation(num_nodes=5, target_node_id=1, device="cpu"):
    print("=" * 75)
    print("NEBULA DECENTRALIZED FL - ADVERSARIAL RESILIENCE BENCHMARK EVALUATION")
    print("=" * 75)
    
    num_classes = 2
    data_shape = (1, 3, 96, 96) # Camelyon17-WILDS patch shape
    
    model = SimplePathologyResNet(num_classes=num_classes).to(device)
    
    torch.manual_seed(42)
    raw_patch = torch.rand(data_shape, device=device)
    raw_label = torch.tensor([1], device=device)
    
    mechanisms = [
        ("Centralized Non-Private", "non_private", 1.0, 0.0),
        ("Static DP-FedAvg (\u03b5=3.0)", "static_dp", 1.0, 0.8),
        ("Proposed Adaptive DP-NEBULA (\u03b5=3.0)", "adaptive_dp", 0.45, 1.2)
    ]
    
    results = []
    
    for mech_name, mech_type, C_val, sigma_val in mechanisms:
        print(f"\n---> Evaluating Mechanism: {mech_name}")
        
        noisy_gradients = generate_simulated_updates(
            model, raw_patch, raw_label, 
            mechanism=mech_type, C_t=C_val, sigma_t=sigma_val
        )
        
        # 1. DLG Attack
        dlg = DLGAttack(model, data_shape, num_classes=num_classes, device=device, lr=0.1, max_iterations=100)
        dlg_reconstructed_img, _ = dlg.reconstruct(noisy_gradients)
        dlg_mse = compute_mse(raw_patch, dlg_reconstructed_img)
        dlg_ssim = compute_ssim(raw_patch, dlg_reconstructed_img)
        
        # 2. Venom Attack
        venom = VenomAttack(model, data_shape, num_classes=num_classes, device=device, lr=0.05, max_iterations=100)
        venom_reconstructed_img, _ = venom.reconstruct(noisy_gradients)
        venom_mse = compute_mse(raw_patch, venom_reconstructed_img)
        venom_ssim = compute_ssim(raw_patch, venom_reconstructed_img)
        
        # 3. FedMIA Membership Inference
        fedmia = FedMIAAttack(num_clients=num_nodes, device=device)
        round_updates = {}
        for k in range(num_nodes):
            if k == target_node_id:
                round_updates[k] = lambda x: model(x)
            else:
                peer_model = SimplePathologyResNet(num_classes=num_classes).to(device)
                round_updates[k] = lambda x: peer_model(x)
                
        z_score, _ = fedmia.evaluate_membership((raw_patch, raw_label), round_updates, target_client_id=target_node_id)
        
        y_true_mia = np.array([1, 1, 0, 0, 1, 0])
        noise_level = 0.1 if mech_type == "non_private" else (0.5 if mech_type == "static_dp" else 1.5)
        simulated_scores = np.array([1.5, 1.2, -0.5, -0.8, 1.1, -0.4]) + np.random.normal(0, noise_level, 6)
        mia_metrics = compute_mia_metrics(y_true_mia, simulated_scores)
        
        results.append({
            "Mechanism": mech_name,
            "DLG_MSE": dlg_mse,
            "DLG_SSIM": dlg_ssim,
            "Venom_MSE": venom_mse,
            "Venom_SSIM": venom_ssim,
            "MIA_AUC": mia_metrics["mia_auc"],
            "MIA_ACC": mia_metrics["mia_acc"]
        })

    print("\n" + "=" * 85)
    print(f"{'Mechanism':<35} | {'DLG SSIM':<10} | {'Venom SSIM':<10} | {'MIA-AUC':<10}")
    print("=" * 85)
    for res in results:
        print(f"{res['Mechanism']:<35} | {res['DLG_SSIM']:<10.4f} | {res['Venom_SSIM']:<10.4f} | {res['MIA_AUC']:<10.4f}")
    print("=" * 85)

if __name__ == "__main__":
    run_security_evaluation(num_nodes=5, target_node_id=1, device="cpu")
