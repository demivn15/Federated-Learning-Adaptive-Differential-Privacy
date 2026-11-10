import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


class StructuralPriorExtractor(nn.Module):
    """
    Structural Prior Extraction (SPE) Module.
    Extracts latent feature representations directly from the gradient matrix
    to guide the reconstruction process away from DP noise traps.
    """
    def __init__(self, feature_dim=512):
        super(StructuralPriorExtractor, self).__init__()
        self.energy_aggregator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, feature_map):
        return self.energy_aggregator(feature_map)


class VenomAttack:
    def __init__(self, model, data_shape, num_classes, device='cpu', 
                 feature_layer_name='backbone.layer4', lr=0.05, max_iterations=100):
        """
        Initializes the Venom Liquid Diffusion-Guided Gradient Inversion Attack.
        
        Args:
            model (nn.Module): Target local architecture (e.g., ResNet-18).
            data_shape (tuple): Shape of the target medical image tensor (1, 3, 96, 96).
            num_classes (int): Number of diagnostic output classes.
            device (str): Computing device ('cpu' or 'cuda').
            feature_layer_name (str): Name of the intermediate layer used for SPE extraction.
            lr (float): Learning rate for latent state optimization.
            max_iterations (int): Maximum optimization steps.
        """
        self.model = deepcopy(model).to(device)
        self.model.eval()
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.max_iterations = max_iterations
        
        # Setup SPE feature hook with robust matching
        self.feature_layer_name = feature_layer_name
        self.extracted_features = None
        self._register_hooks()
        
        self.spe_module = StructuralPriorExtractor(feature_dim=512).to(device)

    def _register_hooks(self):
        """Registers a forward hook to extract latent feature activations robustly."""
        hook_registered = False
        def hook(module, input, output):
            self.extracted_features = output
        
        for name, layer in self.model.named_modules():
            if self.feature_layer_name in name or name.endswith(self.feature_layer_name):
                layer.register_forward_hook(hook)
                hook_registered = True
                break
        
        if not hook_registered:
            # Fallback to last convolutional layer if specific name not found
            for name, layer in reversed(list(self.model.named_modules())):
                if isinstance(layer, nn.Conv2d):
                    layer.register_forward_hook(hook)
                    break

    def _structural_prior_loss(self, dummy_features, target_gradient):
        """Computes energy-based structural matching loss."""
        if dummy_features is None:
            return torch.tensor(0.0, device=self.device)
        dummy_energy = self.spe_module(dummy_features)
        target_norm = torch.norm(target_gradient[-2], p=2) if len(target_gradient) > 1 else torch.norm(target_gradient[0])
        prior_loss = F.mse_loss(dummy_energy, torch.full_like(dummy_energy, target_norm.item()))
        return prior_loss

    def reconstruct(self, target_gradients, diffusion_prior_net=None):
        """
        Executes Venom's Liquid Diffusion-Guided Inversion with smooth sigmoid activation
        to prevent flat gradient regions during clamping.
        """
        if diffusion_prior_net is not None:
            latent_z = torch.randn((1, 128), device=self.device, requires_grad=True)
            optimizer = optim.Adam([latent_z], lr=self.lr)
        else:
            # Initialize with small normal noise for smooth sigmoid scaling
            dummy_latent = torch.randn(self.data_shape, device=self.device, requires_grad=True) * 0.1
            optimizer = optim.Adam([dummy_latent], lr=self.lr)

        dummy_label = torch.randn((1, self.num_classes), device=self.device, requires_grad=True)
        label_optimizer = optim.Adam([dummy_label], lr=self.lr)

        for iters in range(self.max_iterations):
            optimizer.zero_grad()
            label_optimizer.zero_grad()

            if diffusion_prior_net is not None:
                current_image = diffusion_prior_net(latent_z)
            else:
                # Use sigmoid to map unconstrained latent space to [0, 1] without flat clamping gradients
                current_image = torch.sigmoid(dummy_latent)

            pred = self.model(current_image)
            dummy_loss = F.cross_entropy(pred, F.softmax(dummy_label, dim=-1))

            dummy_grads = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

            grad_matching_loss = 0.0
            for dg, tg in zip(dummy_grads, target_gradients):
                if tg is not None:
                    grad_matching_loss += F.mse_loss(dg, tg)

            spe_loss = self._structural_prior_loss(self.extracted_features, target_gradients)

            total_loss = grad_matching_loss + 0.1 * spe_loss
            total_loss.backward()

            optimizer.step()
            label_optimizer.step()

        final_image = diffusion_prior_net(latent_z).detach() if diffusion_prior_net is not None else torch.sigmoid(dummy_latent).detach()
        reconstructed_label = torch.argmax(dummy_label, dim=-1).detach()

        return final_image, reconstructed_label
