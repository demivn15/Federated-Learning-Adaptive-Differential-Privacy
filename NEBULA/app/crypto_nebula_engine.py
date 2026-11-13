import os
import time
import math
import struct
import hashlib
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.exceptions import InvalidSignature


class RenyiDPAccountant:
    """
    Contador analítico Rényi DP (RDP) bajo submuestreo de Poisson con filtro individual.
    Cumple con las Ecuaciones (5) y (6) del artículo IEEE Access.
    """
    def __init__(self, sampling_rate_q: float = 0.05, delta: float = 1e-5, max_epsilon: float = 3.0):
        self.q = sampling_rate_q
        self.delta = delta
        self.max_epsilon = max_epsilon
        self.orders = np.arange(2.0, 65.0, 1.0)
        self.rdp_budget = np.zeros_like(self.orders, dtype=np.float64)
        self.history: List[Dict[str, Any]] = []

    def _poisson_subsampled_rdp(self, sigma: float) -> np.ndarray:
        step = np.zeros_like(self.orders, dtype=np.float64)
        for idx, alpha in enumerate(self.orders):
            if sigma <= 0:
                step[idx] = np.inf
                continue
            standard_rdp = alpha / (2.0 * (sigma ** 2))
            amplified_rdp = 2.0 * (self.q ** 2) * alpha / (sigma ** 2)
            step[idx] = min(standard_rdp, amplified_rdp) if sigma >= 0.5 else standard_rdp
        return step

    def step(self, sigma_t: float, round_t: int) -> Tuple[float, float, bool]:
        self.rdp_budget += self._poisson_subsampled_rdp(sigma_t)
        eps_candidates = self.rdp_budget + np.log(1.0 / self.delta) / (self.orders - 1.0)
        best_idx = int(np.argmin(eps_candidates))
        eps_val = float(eps_candidates[best_idx])
        best_alpha = float(self.orders[best_idx])
        passed = eps_val <= self.max_epsilon

        self.history.append({
            "round": round_t,
            "sigma_t": round(sigma_t, 5),
            "epsilon": round(eps_val, 5),
            "optimal_alpha": best_alpha,
            "privacy_filter_ok": passed
        })
        return eps_val, best_alpha, passed


class AdaptiveSensitivityEngine:
    """
    Motor de acotamiento dinámico de sensibilidad L2 (C_t) y ruido gaussiano
    calibrado mediante entropía de Shannon normalizada.
    """
    def __init__(self, c0: float = 1.0, gamma: float = 0.05, lam: float = 0.2,
                 sigma_base: float = 0.85, eta: float = 0.5):
        self.c0 = c0
        self.gamma = gamma
        self.lam = lam
        self.sigma_base = sigma_base
        self.eta = eta

    def compute_dynamic_c_t(self, grad_norms: torch.Tensor, round_t: int) -> float:
        var_t = float(torch.var(grad_norms).item()) if grad_norms.numel() > 1 else 0.0
        c_t = self.c0 * math.exp(-self.gamma * round_t) + self.lam * math.sqrt(max(var_t, 0.0))
        return float(max(c_t, 1e-4))

    @staticmethod
    def normalized_shannon_entropy(tensor: torch.Tensor) -> float:
        flat = torch.abs(tensor.detach().flatten()).float()
        l1_norm = float(torch.sum(flat).item()) + 1e-12
        probs = torch.clamp(flat / l1_norm, min=1e-12)
        entropy = float(-torch.sum(probs * torch.log(probs)).item())
        max_entropy = math.log(max(flat.numel(), 2))
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def clip_and_perturb(self, grad: torch.Tensor, c_t: float) -> Tuple[torch.Tensor, float, float, float]:
        l2_norm = float(torch.norm(grad, p=2).item())
        clip_factor = max(1.0, l2_norm / c_t)
        clipped = grad / clip_factor
        h_norm = self.normalized_shannon_entropy(clipped)
        sigma_t = self.sigma_base * (1.0 + self.eta * (1.0 - h_norm))
        noise = torch.randn_like(clipped) * (sigma_t * c_t)
        return clipped + noise, sigma_t, h_norm, l2_norm


class NebulaCryptoHospitalNode:
    """
    Nodo Hospitalario P2P para NEBULA:
    - Acuerdo de claves X25519 (Curve25519 ECDH)
    - Derivación HKDF-SHA256
    - Enmascaramiento aditivo IND-CPA con AES-256-CTR en Z_{2^32}
    - Firma digital Ed25519 por ronda
    """
    MODULO = 1 << 32
    SCALE = 1 << 16

    def __init__(self, node_id: int, dp_engine: AdaptiveSensitivityEngine,
                 sampling_q: float = 0.05, delta: float = 1e-5, target_eps: float = 2.0):
        self.node_id = node_id
        self.dp_engine = dp_engine
        self.accountant = RenyiDPAccountant(sampling_q, delta, target_eps)

        self.dh_private = x25519.X25519PrivateKey.generate()
        self.dh_public = self.dh_private.public_key()
        self.sig_private = ed25519.Ed25519PrivateKey.generate()
        self.sig_public = self.sig_private.public_key()
        self.pairwise_seeds: Dict[int, bytes] = {}

    def export_public_keys(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dh_public": self.dh_public,
            "sig_public": self.sig_public,
            "dh_hex": self.dh_public.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            ).hex(),
            "sig_hex": self.sig_public.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            ).hex(),
        }

    def establish_ecdh_session(self, peers_registry: Dict[int, Dict[str, Any]]):
        for peer_id, bundle in peers_registry.items():
            if peer_id == self.node_id:
                continue
            shared_secret = self.dh_private.exchange(bundle["dh_public"])
            low, high = min(self.node_id, peer_id), max(self.node_id, peer_id)
            info_tag = f"NEBULA-P2P-SECAGG-{low}-{high}".encode("utf-8")
            self.pairwise_seeds[peer_id] = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=info_tag
            ).derive(shared_secret)

    @staticmethod
    def _generate_aes_ctr_mask(seed: bytes, round_t: int, num_elements: int) -> np.ndarray:
        nonce = struct.pack(">QQ", 0x4E454255, round_t)
        encryptor = Cipher(algorithms.AES(seed), modes.CTR(nonce)).encryptor()
        keystream = encryptor.update(b"\x00" * (num_elements * 4)) + encryptor.finalize()
        return np.frombuffer(keystream, dtype=np.uint32).astype(np.int64)

    @staticmethod
    def byte_shannon_entropy(byte_stream: bytes) -> float:
        arr = np.frombuffer(byte_stream, dtype=np.uint8)
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / arr.size
        return float(-np.sum(probs * np.log2(probs)))

    def protect_gradient_round(self, grad: torch.Tensor, c_t: float, round_t: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        noisy_grad, sigma_t, h_grad, raw_norm = self.dp_engine.clip_and_perturb(grad, c_t)
        eps_t, alpha_opt, filter_ok = self.accountant.step(sigma_t, round_t)

        flat_np = noisy_grad.detach().cpu().numpy().flatten()
        quantized = np.mod(np.round(flat_np * self.SCALE).astype(np.int64), self.MODULO)
        masked = quantized.copy()

        for peer_id, seed in self.pairwise_seeds.items():
            mask = self._generate_aes_ctr_mask(seed, round_t, quantized.size)
            if self.node_id < peer_id:
                masked = np.mod(masked + mask, self.MODULO)
            else:
                masked = np.mod(masked - mask, self.MODULO)

        ct_bytes = masked.astype(np.uint32).tobytes()
        digest = hashlib.sha256(ct_bytes).digest()
        header = struct.pack(">II", round_t, self.node_id) + digest
        signature = self.sig_private.sign(header)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Representación de punto flotante que observa un atacante externo al interceptar c_t^k
        signed_masked = np.where(masked >= (self.MODULO // 2), masked - self.MODULO, masked)
        intercepted_tensor = torch.tensor(
            signed_masked.astype(np.float32) / self.SCALE
        ).reshape(grad.shape)

        # Correlación de Pearson entre el gradiente original y el criptograma interceptado
        orig_flat = grad.detach().cpu().numpy().flatten()
        cipher_flat = signed_masked.astype(np.float64)
        if np.std(orig_flat) > 1e-12 and np.std(cipher_flat) > 1e-12:
            corr = float(np.corrcoef(orig_flat, cipher_flat)[0, 1])
        else:
            corr = 0.0

        return {
            "node_id": self.node_id,
            "round_t": round_t,
            "ciphertext": masked,
            "ciphertext_bytes": ct_bytes,
            "signature": signature,
            "cleartext_noisy_grad": noisy_grad,
            "intercepted_tensor": intercepted_tensor,
            "metrics": {
                "raw_l2_norm": round(raw_norm, 5),
                "dynamic_c_t": round(c_t, 5),
                "grad_shannon_entropy": round(h_grad, 5),
                "adaptive_sigma_t": round(sigma_t, 5),
                "cumulative_epsilon": round(eps_t, 5),
                "optimal_alpha": alpha_opt,
                "privacy_filter_ok": filter_ok,
                "ciphertext_byte_entropy": round(self.byte_shannon_entropy(ct_bytes), 5),
                "plaintext_ciphertext_corr": round(corr, 6),
                "crypto_latency_ms": round(elapsed_ms, 3),
                "payload_kb": round(len(ct_bytes) / 1024.0, 2)
            }
        }


class NebulaP2PAggregator:
    """
    Verificador de firmas Ed25519 y agregador homomórfico modular sobre el anillo Z_{2^32}.
    """
    @staticmethod
    def verify_and_aggregate(packets: List[Dict[str, Any]],
                             peers_registry: Dict[int, Dict[str, Any]],
                             original_shape: tuple) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t0 = time.perf_counter()
        mod = NebulaCryptoHospitalNode.MODULO
        scale = NebulaCryptoHospitalNode.SCALE
        k_nodes = len(packets)

        agg_mod = np.zeros_like(packets[0]["ciphertext"], dtype=np.int64)
        verified_count = 0

        for pkt in packets:
            nid = pkt["node_id"]
            digest = hashlib.sha256(pkt["ciphertext_bytes"]).digest()
            header = struct.pack(">II", pkt["round_t"], nid) + digest
            peers_registry[nid]["sig_public"].verify(pkt["signature"], header)
            agg_mod = np.mod(agg_mod + pkt["ciphertext"], mod)
            verified_count += 1

        signed_sum = np.where(agg_mod >= (mod // 2), agg_mod - mod, agg_mod)
        recovered_mean = torch.tensor(
            signed_sum.astype(np.float32) / (scale * k_nodes)
        ).reshape(original_shape)

        true_mean = torch.stack([p["cleartext_noisy_grad"] for p in packets]).mean(dim=0)
        cancellation_mse = float(torch.mean((recovered_mean - true_mean) ** 2).item())
        agg_ms = (time.perf_counter() - t0) * 1000.0

        # Prueba de detección de adulteración activa (modificando 1 bit en tránsito)
        tampered = bytearray(packets[0]["ciphertext_bytes"])
        tampered[0] ^= 0x01
        tamper_detected = False
        try:
            bad_digest = hashlib.sha256(bytes(tampered)).digest()
            bad_header = struct.pack(">II", packets[0]["round_t"], packets[0]["node_id"]) + bad_digest
            peers_registry[packets[0]["node_id"]]["sig_public"].verify(packets[0]["signature"], bad_header)
        except InvalidSignature:
            tamper_detected = True

        return recovered_mean, {
            "verified_signatures": f"{verified_count}/{k_nodes}",
            "tamper_attack_blocked": tamper_detected,
            "modular_cancellation_mse": cancellation_mse,
            "aggregation_time_ms": round(agg_ms, 3)
        }