import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


class Day3ChecklistVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m_file = ROOT_DIR / "results" / "master_publication_results.json"
        assert m_file.exists(), "Falta master_publication_results.json"
        cls.master = json.loads(m_file.read_text(encoding="utf-8"))

    def test_task_3_1_inversion_and_fedmia_bounds(self):
        for ck_k, ck_v in self.master["day1"]["checkpoint_attacks"].items():
            self.assertGreater(ck_v["Unprotected"]["dlg_ssim"], 0.85)
            self.assertLess(ck_v["Proposed_Adaptive_DP_NEBULA"]["dlg_ssim"], 0.20)
            self.assertLess(ck_v["Proposed_Adaptive_DP_NEBULA"]["venom_ssim"], 0.20)

    def test_task_3_2_ablation_and_resnet18_crypto(self):
        abl = self.master["day3"]["exp4_component_ablation"]
        self.assertEqual(len(abl), 4)
        rn = self.master["day3"]["exp4_resnet18_44mb_crypto_overhead"]
        self.assertEqual(rn["gradient_payload_size_mb"], 44.0)
        self.assertLess(rn["total_per_round_crypto_overhead_ms"], 500.0)

    def test_task_3_3_latex_tables_exist(self):
        for t_name in [
            "table1_privacy_utility.tex",
            "table2_non_iid_dirichlet.tex",
            "table3_inversion_checkpoints.tex",
            "table4_ablation_and_crypto.tex",
        ]:
            p = ROOT_DIR / "results" / "tables" / t_name
            self.assertTrue(p.exists(), f"Falta tabla {t_name}")
            self.assertGreater(p.stat().st_size, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
