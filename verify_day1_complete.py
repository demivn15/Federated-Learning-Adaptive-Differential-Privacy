import json
import unittest
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent


class Day1ChecklistVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache_pt = ROOT_DIR / "data" / "camelyon17_wilds" / "camelyon17_wilds_96x96_5hospitals.pt"
        assert cache_pt.exists(), "Falta el archivo de parches reales Camelyon17-WILDS."
        cls.blob = torch.load(cache_pt, weights_only=True)

        master_json = ROOT_DIR / "results" / "master_publication_results.json"
        assert master_json.exists(), "Falta master_publication_results.json."
        cls.master = json.loads(master_json.read_text(encoding="utf-8"))
        cls.d1 = cls.master["day1"]

    def test_task_1_1_real_camelyon17_and_public_calibration_split(self):
        """Verifica parches reales 96x96x3 en 5 hospitales y split público entre 2% y 5%."""
        self.assertEqual(self.blob.get("source"), "REAL_CAMELYON17_WILDS_HUGGINGFACE")
        self.assertEqual(tuple(self.blob["patches"].shape[1:]), (3, 96, 96))
        self.assertEqual(len(torch.unique(self.blob["centers"])), 5)
        calib_pct = self.d1["public_calibration_percentage"]
        self.assertGreaterEqual(calib_pct, 2.0)
        self.assertLessEqual(calib_pct, 5.0)

    def test_task_1_1_per_example_clipping_and_exact_rdp_budgets(self):
        """Verifica decaimiento de C_t y cierre exacto de Rényi DP en eps in {1.0, 2.0, 3.0}."""
        traj = self.d1["round_trajectory"]
        self.assertGreater(traj[0]["c_t"], traj[-1]["c_t"])
        for eps_key, target in [("epsilon_1.0", 1.0), ("epsilon_2.0", 2.0), ("epsilon_3.0", 3.0)]:
            achieved = self.d1["rdp_exact_calibrations"][eps_key]["achieved_epsilon_T"]
            self.assertAlmostEqual(achieved, target, places=3)

    def test_task_1_2_authenticated_p2p_crypto_and_gossip(self):
        """Verifica X25519 + HKDF-SHA256 + AES-256-CTR + Ed25519 y entropía > 7.99 bits/B."""
        crypto_stats = self.d1["p2p_crypto_gossip_timings_ms"]
        self.assertGreater(crypto_stats["avg_byte_entropy"], 7.99)
        self.assertEqual(len(self.d1["latest_packets_preview"]), 5)

    def test_task_1_3_checkpoints_and_attack_bounds(self):
        """Verifica checkpoints t=1,10,25 y que DLG y Venom cumplan SSIM < 0.20 bajo DP-NEBULA."""
        for r in [1, 10, 25]:
            ckpt_file = ROOT_DIR / "checkpoints" / f"checkpoint_t{r}.pt"
            self.assertTrue(ckpt_file.exists(), f"Falta {ckpt_file}")
            atk = self.d1["checkpoint_attacks"][f"t_{r}"]
            self.assertGreater(atk["Unprotected"]["dlg_ssim"], 0.85)
            self.assertLess(atk["Proposed_Adaptive_DP_NEBULA"]["dlg_ssim"], 0.20)
            self.assertLess(atk["Proposed_Adaptive_DP_NEBULA"]["venom_ssim"], 0.20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
