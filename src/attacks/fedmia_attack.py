import torch
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

    def _compute_measurement(self, target_sample, client_update):
        """
        Step 1: Computes the low-dimensional measurement M(I | (x, y)) 
        based on loss or gradient alignment metrics for the target sample.
        """
        x, y = target_sample
        # Example measurement: Cross-entropy loss or gradient norm signature on the update
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            # Evaluation of sample loss under the client's shared parameter state
            loss_val = criterion(client_update(x), y).item()
        return loss_val

    def evaluate_membership(self, target_sample, round_updates_dict, target_client_id):
        """
        Steps 2 & 3: Estimates the out-distribution from non-target peer updates 
        and executes the one-tailed likelihood-ratio test across T rounds.
        
        Args:
            target_sample (tuple): (x, y) patient histopathology patch and label.
            round_updates_dict (dict): Dictionary mapping client_id to their model updates for round t.
                                      Format: {k: update_tensor_or_model for k in range(K)}
            target_client_id (int): ID of the hospital node being audited (e.g., node 1).
            
        Returns:
            float: Likelihood ratio test score / membership score.
            bool: Binary inference decision (True = Member, False = Non-Member).
        """
        # Collect measurements from non-target clients (exploiting the "all for one" principle)
        out_measurements = []
        target_measurement = None

        for k, update in round_updates_dict.items():
            # Mocking client update application or state evaluation function
            measurement = self._compute_measurement(target_sample, update)
            
            if k == target_client_id:
                target_measurement = measurement
            else:
                out_measurements.append(measurement)

        if not out_measurements or target_measurement is None:
            raise ValueError("Invalid client update dictionary or target client ID provided.")

        # Step 2: Estimate Q_out distribution parameters from non-target updates
        mu_out = np.mean(out_measurements)
        std_out = np.std(out_measurements) + 1e-8 # Prevent zero division

        # Step 3: One-tailed Likelihood-Ratio Hypothesis Test
        # Evaluates probability density of target measurement against out-distribution
        z_score = (target_measurement - mu_out) / std_out
        p_value = 1.0 - norm.cdf(z_score)

        # Decision threshold (e.g., p-value < 0.05 indicates rejection of non-member null hypothesis)
        is_member = p_value < 0.05

        return float(z_score), bool(is_member)

    def aggregate_multi_round_inference(self, target_sample, historical_rounds_updates, target_client_id):
        """
        Extends FedMIA across T communication rounds to capture temporal information,
        following Theorem 1 of the attack formulation.
        """
        round_scores = []
        inferences = []

        for t, round_updates in enumerate(historical_rounds_updates):
            score, inferred = self.evaluate_membership(target_sample, round_updates, target_client_id)
            round_scores.append(score)
            inferences.append(inferred)

        # Worst-case membership leakage: inferred as member in at least one communication round
        final_membership_decision = any(inferences)
        aggregate_score = np.max(round_scores)

        return aggregate_score, final_membership_decision
