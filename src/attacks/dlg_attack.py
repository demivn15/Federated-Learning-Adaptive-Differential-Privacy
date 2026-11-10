import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

class DLGAttack:
    def __init__(self, model, data_shape, num_classes, device='cpu', 
                 optimizer_type='LBFGS', lr=0.1, max_iterations=300):
        """
        Initializes the Deep Leakage from Gradients (DLG) attack module.
        
        Args:
            model (nn.Module): The local model architecture (e.g., ResNet-18).
            data_shape (tuple): Shape of the target data (e.g., (1, 3, 96, 96)).
            num_classes (int): Number of classification classes.
            device (str): Compute device ('cpu' or 'cuda').
            optimizer_type (str): Optimization algorithm ('LBFGS' or 'Adam').
            lr (float): Learning rate for the dummy data optimization.
            max_iterations (int): Maximum steps for gradient matching.
        """
        self.model = deepcopy(model).to(device)
        self.model.eval() # Ensure dropout/batchnorm are frozen during attack
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.max_iterations = max_iterations

    def _gradient_matching_loss(self, dummy_gradients, target_gradients):
        """Calculates the L2 distance between dummy and intercepted gradients."""
        loss = 0
        for dummy_g, target_g in zip(dummy_gradients, target_gradients):
            if target_g is not None:
                loss += ((dummy_g - target_g) ** 2).sum()
        return loss

    def reconstruct(self, target_gradients):
        """
        Executes the optimization loop to reconstruct data from gradients.
        
        Args:
            target_gradients (list): Intercepted gradients \tilde{g}_t^k.
            
        Returns:
            tuple: (reconstructed_image, reconstructed_label)
        """
        # 1. Initialize dummy data and dummy label with Gaussian noise
        dummy_data = torch.randn(self.data_shape, device=self.device, requires_grad=True)
        dummy_label = torch.randn((1, self.num_classes), device=self.device, requires_grad=True)

        # 2. Setup the optimizer for the dummy variables
        if self.optimizer_type == 'LBFGS':
            optimizer = optim.LBFGS([dummy_data, dummy_label], lr=self.lr)
        else:
            optimizer = optim.Adam([dummy_data, dummy_label], lr=self.lr)

        # 3. Optimization Loop
        for iters in range(self.max_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Forward pass dummy data through the frozen model
                dummy_pred = self.model(dummy_data)
                
                # Calculate dummy loss (e.g., CrossEntropy approximated for soft labels)
                dummy_loss = F.cross_entropy(dummy_pred, F.softmax(dummy_label, dim=-1))
                
                # Compute dummy gradients
                dummy_gradients = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                
                # Calculate gradient matching loss
                match_loss = self._gradient_matching_loss(dummy_gradients, target_gradients)
                
                match_loss.backward()
                return match_loss

            optimizer.step(closure)

            if iters % 50 == 0:
                current_loss = closure().item()
                print(f"DLG Iteration {iters} | Gradient Matching Loss: {current_loss:.4f}")

        # Post-process reconstructed label to one-hot or argmax
        reconstructed_label = torch.argmax(dummy_label, dim=-1)
        
        return dummy_data.detach(), reconstructed_label.detach()
