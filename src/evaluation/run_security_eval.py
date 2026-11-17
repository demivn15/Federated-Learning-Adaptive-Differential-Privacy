import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.benchmarks import execute_full_project_pipeline


def main():
    print("=== EJECUTANDO ARNÉS DE SEGURIDAD DIA 1 (run_security_eval.py) ===")
    master = execute_full_project_pipeline()
    d1 = master["day1"]

    day1_export = {
        "status": "DAY_1_COMPLETED_AND_VERIFIED",
        "task_1_1_fl_and_rdp_engine": {
            "dataset": "Camelyon17-WILDS (Real 96x96x3 H&E Patches, K=5 Hospitals)",
            "hospital_counts": d1["data_summary"],
            "public_calibration_percentage": d1["public_calibration_percentage"],
            "public_calibration_samples": d1["public_calibration_samples"],
            "rdp_exact_calibrations": d1["rdp_exact_calibrations"],
            "final_round_25_state": d1["round_trajectory"][-1],
        },
        "task_1_2_authenticated_p2p_crypto_gossip": {
            "primitives": "Ed25519 (RFC 8032) + X25519 ECDH (RFC 7748) + HKDF-SHA256 + AES-256-CTR",
            "scheme": "Encrypt-then-Sign (ciphertext || round_id || sender_id)",
            "average_timings_and_entropy": d1["p2p_crypto_gossip_timings_ms"],
            "nodes_snapshot": d1["latest_packets_preview"],
        },
        "task_1_3_real_checkpoints_attack_harness": {
            "checkpoints_evaluated": ["checkpoint_t1.pt", "checkpoint_t10.pt", "checkpoint_t25.pt"],
            "dlg_300_and_venom_100_metrics": d1["checkpoint_attacks"],
            "fedmia_trajectory_summary": [
                d1["fedmia_trajectory"][0],
                d1["fedmia_trajectory"][9],
                d1["fedmia_trajectory"][24],
            ],
        },
    }

    out_path = ROOT_DIR / "results" / "day1" / "day1_security_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(day1_export, indent=2), encoding="utf-8")
    print(f"Reporte oficial del Día 1 exportado en: {out_path}")


if __name__ == "__main__":
    main()
