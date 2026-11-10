import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

class FedMIAAttack:
    def __init__(self, num_clients=5, device='cpu'):
        """
        Initializes the FedMIA (Membership Inference Attack) module using 
        the 'all for one' multi-round likelihood-ratio approach.
        
        Args:
            num_clients (int): Total number of decentralized hospital nodes (K=5).
            device (str): Computing device ('cpu' or 'cuda').
        """
        self.num_clients = num_clients
        self.device = device

    def _compute_measurement(self, target_sample, client_model):
        """
        Step 1: Computes the low-dimensional measurement M(I | (x, y)) 
        based on loss for the target sample under the given client model state.
        """
        x, y = target_sample
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            if isinstance(client_model, nn.Module):
                client_model.eval()
                outputs = client_model(x)
            elif callable(client_model):
                outputs = client_model(x)
            else:
                raise TypeError("Client update/model must be an nn.Module or callable.")
            loss_val = criterion(outputs, y).item()
        return loss_val

    def evaluate_membership(self, target_sample, round_updates_dict, target_client_id):
        """
        Steps 2 & 3: Estimates the out-distribution from non-target peer updates 
        and executes the one-tailed likelihood-ratio test (lower loss for members).
        
        Args:
            target_sample (tuple): (x, y) patient histopathology patch and label.
            round_updates_dict (dict): Dictionary mapping client_id to their model/update for round t.
            target_client_id (int): ID of the hospital node being audited.
            
        Returns:
            tuple: (z_score, is_member)
        """
        out_measurements = []
        target_measurement = None

        for k, client_model in round_updates_dict.items():
            measurement = self._compute_measurement(target_sample, client_model)
            
            if k == target_client_id:
                target_measurement = measurement
            else:
                out_measurements.append(measurement)

        if not out_measurements or target_measurement is None:
            raise ValueError("Invalid client update dictionary or target client ID provided.")

        mu_out = np.mean(out_measurements)
        std_out = np.std(out_measurements) + 1e-8 # Prevent zero division

        # Members typically exhibit lower loss on training data than non-members.
        # Z-score measures how many standard deviations the target loss is below the out-distribution mean.
        z_score = (mu_out - target_measurement) / std_out
        
        # One-tailed p-value for lower loss (membership hypothesis)
        p_value = 1.0 - norm.cdf(z_score)
        is_member = p_value < 0.05

        return float(z_score), bool(is_member)

    def aggregate_multi_round_inference(self, target_sample, historical_rounds_updates, target_client_id):
        """
        Extends FedMIA across T communication rounds to capture temporal information.
        """
        round_scores = []
        inferences = []

        for round_updates in historical_rounds_updates:
            score, inferred = self.evaluate_membership(target_sample, round_updates, target_client_id)
            round_scores.append(score)
            inferences.append(inferred)

        # Worst-case membership leakage: inferred as member in at least one communication round
        final_membership_decision = any(inferences)
        aggregate_score = float(np.max(round_scores))

        return aggregate_score, final_membership_decision
