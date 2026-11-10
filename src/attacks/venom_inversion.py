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
                 feature_layer_name='layer4', lr=0.05, max_iterations=200):
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
        
        # Setup SPE feature hook
        self.feature_layer_name = feature_layer_name
        self.extracted_features = None
        self._register_hooks()
        
        # Instantiate SPE helper module
        # Assumes standard ResNet feature dimension (e.g., 512 for ResNet-18 layer4)
        self.spe_module = StructuralPriorExtractor(feature_dim=512).to(device)

    def _register_hooks(self):
        """Registers a forward hook to extract latent feature activations."""
        def hook(module, input, output):
            self.extracted_features = output
        
        for name, layer in self.model.named_modules():
            if name == self.feature_layer_name:
                layer.register_forward_hook(hook)
                break

    def _structural_prior_loss(self, dummy_features, target_gradient):
        """
        Computes energy-based structural matching loss between dummy feature 
        maps and the energy signature of the intercepted gradient update.
        """
        dummy_energy = self.spe_module(dummy_features)
        # Approximate target gradient norm signature as energy prior target
        target_norm = torch.norm(target_gradient[-2], p=2) if len(target_gradient) > 1 else torch.norm(target_gradient[0])
        prior_loss = F.mse_loss(dummy_energy, torch.full_like(dummy_energy, target_norm.item()))
        return prior_loss

    def reconstruct(self, target_gradients, diffusion_prior_net=None):
        """
        Executes Venom's Liquid Diffusion-Guided Inversion.
        
        Args:
            target_gradients (list): Intercepted DP-perturbed updates \tilde{g}_t^k.
            diffusion_prior_net (nn.Module, optional): Pre-trained generative/diffusion 
                                                       prior for guided manifold sampling.
            
        Returns:
            tuple: (reconstructed_image, reconstructed_label)
        """
        # 1. Initialize latent representation z (or image directly if diffusion prior is None)
        if diffusion_prior_net is not None:
            latent_z = torch.randn((1, 128), device=self.device, requires_grad=True)
            optimizer = optim.Adam([latent_z], lr=self.lr)
        else:
            dummy_data = torch.randn(self.data_shape, device=self.device, requires_grad=True)
            optimizer = optim.Adam([dummy_data], lr=self.lr)

        dummy_label = torch.randn((1, self.num_classes), device=self.device, requires_grad=True)
        label_optimizer = optim.Adam([dummy_label], lr=self.lr)

        # 2. Diffusion-Guided Recovery Optimization Loop
        for iters in range(self.max_iterations):
            optimizer.zero_grad()
            label_optimizer.zero_grad()

            # Map latent z to image via Diffusion/Generative prior if available
            if diffusion_prior_net is not None:
                current_image = diffusion_prior_net(latent_z)
            else:
                current_image = dummy_data

            # Clamp intermediate generation to valid normalized bounds [0, 1]
            current_image_clamped = torch.clamp(current_image, 0.0, 1.0)

            # Forward pass through model
            pred = self.model(current_image_clamped)
            dummy_loss = F.cross_entropy(pred, F.softmax(dummy_label, dim=-1))

            # Compute dummy gradients
            dummy_grads = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

            # Primary Gradient Matching Loss
            grad_matching_loss = 0.0
            for dg, tg in zip(dummy_grads, target_gradients):
                if tg is not None:
                    grad_matching_loss += F.mse_loss(dg, tg)

            # Structural Prior Energy Loss (SPE component)
            spe_loss = 0.0
            if self.extracted_features is not None:
                spe_loss = self._structural_prior_loss(self.extracted_features, target_gradients)

            # Total Guided Loss
            total_loss = grad_matching_loss + 0.1 * spe_loss
            total_loss.backward()

            optimizer.step()
            label_optimizer.step()

            if iters % 50 == 0:
                print(f"Venom Iteration {iters:03d} | Match Loss: {grad_matching_loss.item():.4f} | SPE Loss: {spe_loss.item() if isinstance(spe_loss, torch.Tensor) else spe_loss:.4f}")

        # Final reconstructed output
        final_image = diffusion_prior_net(latent_z).detach() if diffusion_prior_net is not None else dummy_data.detach()
        reconstructed_label = torch.argmax(dummy_label, dim=-1).detach()

        return torch.clamp(final_image, 0.0, 1.0), reconstructed_label
