import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def export_latex_tables_from_json(master: dict):
    """Task 3.3: Exporta automáticamente las 4 tablas LaTeX (.tex) en results/tables/."""
    t_dir = ROOT_DIR / "results" / "tables"
    t_dir.mkdir(parents=True, exist_ok=True)
    nl = chr(10)

    # Tabla 1: Experimento 1 (5 Semillas Privacidad-Utilidad)
    t1 = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Experiment 1: Diagnostic Utility Across 5 Seeds on Camelyon17-WILDS ($\delta=10^{-5}$).}",
        r"\label{tab:exp1}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Method} & $\boldsymbol{\varepsilon}$ & \textbf{Test AUROC} & \textbf{Sensitivity} & \textbf{Macro-F1} \\",
        r"\midrule",
    ]
    for r in master["day2"]["exp1_privacy_utility_table"]:
        m_name = r["method"].replace("&", r"\&")
        eps_s = r"$\infty$" if r["eps"] == "inf" else r["eps"]
        auc_s = r["auroc"].replace("±", r"\pm")
        sen_s = r["sens"].replace("±", r"\pm").replace("%", r"\%")
        f1_s = r["f1"].replace("±", r"\pm")
        if "Proposed" in m_name:
            t1.append(f"\\textbf{{{m_name}}} & $\\mathbf{{{eps_s}}}$ & $\\mathbf{{{auc_s}}}$ & $\\mathbf{{{sen_s}}}$ & $\\mathbf{{{f1_s}}}$ \\\\")
        else:
            t1.append(f"{m_name} & ${eps_s}$ & ${auc_s}$ & ${sen_s}$ & ${f1_s}$ \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (t_dir / "table1_privacy_utility.tex").write_text(nl.join(t1), encoding="utf-8")

    # Tabla 2: Experimento 2 (Dirichlet Non-IID & Client Drift)
    e2 = master["day2"].get("exp2_non_iid_dirichlet") or master["day2"].get("exp2_non_iid")
    t2 = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Experiment 2: Non-IID Dirichlet Stress Test Across 5 Hospitals ($\varepsilon=2.0$).}",
        r"\label{tab:exp2}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Regime} & \textbf{Method} & \textbf{Mean AUROC} & \textbf{Hosp. Range} & \textbf{Drift $\|\Delta w\|_2$} \\",
        r"\midrule",
    ]
    for b_key, m_dict in e2.items():
        b_lbl = b_key.replace("dirichlet_beta_", r"$\beta = ") + "$"
        for m_code, vals in m_dict.items():
            h_list = vals["per_hospital_auroc"]
            h_rng = f"{min(h_list):.3f}--{max(h_list):.3f}"
            m_clean = m_code.replace("_", " ")
            if "Proposed" in m_code:
                t2.append(f"{b_lbl} & \\textbf{{{m_clean}}} & $\\mathbf{{{vals['mean_auroc']:.4f}}}$ & $\\mathbf{{{h_rng}}}$ & $\\mathbf{{{vals['final_client_drift_l2']:.4f}}}$ \\\\")
            else:
                t2.append(f"{b_lbl} & {m_clean} & ${vals['mean_auroc']:.4f}$ & ${h_rng}$ & ${vals['final_client_drift_l2']:.4f}$ \\\\")
    t2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (t_dir / "table2_non_iid_dirichlet.tex").write_text(nl.join(t2), encoding="utf-8")

    # Tabla 3: Experimento 3 (Checkpoints Reales t=1, 10, 25)
    t3 = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Experiment 3: Gradient Inversion Resilience Across Real Checkpoints ($96\times 96\times 3$).}",
        r"\label{tab:exp3}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Ckpt.} & \textbf{Unprotected DLG} & \textbf{Static DP DLG} & \textbf{Ours DLG} & \textbf{Ours Venom} \\",
        r"\midrule",
    ]
    for ck_k, ck_v in master["day1"]["checkpoint_attacks"].items():
        r_lbl = ck_k.replace("t_", "$t = ") + "$"
        u_s = f"{ck_v['Unprotected']['dlg_ssim']:.4f}"
        st_s = f"{ck_v['Static_DP_eps3']['dlg_ssim']:.4f}"
        pr_d = f"{ck_v['Proposed_Adaptive_DP_NEBULA']['dlg_ssim']:.4f}"
        pr_v = f"{ck_v['Proposed_Adaptive_DP_NEBULA']['venom_ssim']:.4f}"
        t3.append(f"{r_lbl} & ${u_s}$ & ${st_s}$ & $\\mathbf{{{pr_d}}}$ & $\\mathbf{{{pr_v}}}$ \\\\")
    t3 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (t_dir / "table3_inversion_checkpoints.tex").write_text(nl.join(t3), encoding="utf-8")

    # Tabla 4: Experimento 4 (Ablación + ResNet-18 44 MB Crypto Overhead)
    rn = master["day3"]["exp4_resnet18_44mb_crypto_overhead"]
    t4 = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Experiment 4: Component Ablation ($\varepsilon=2.0$) and ResNet-18 (44 MB) Crypto Overhead.}",
        r"\label{tab:exp4}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Ablation Variant} & \textbf{Conv. Round} & \textbf{AUROC} & \textbf{Sens.} & \textbf{DLG SSIM} \\",
        r"\midrule",
    ]
    for ab in master["day3"]["exp4_component_ablation"]:
        v_name = ab["variant"].replace("σ", r"$\sigma$").replace("C_t", r"$C_t$")
        auc_a = ab["auroc_eps2"].replace("±", r"\pm")
        sen_a = ab["sens_eps2"].replace("%", r"\%")
        t4.append(f"{v_name} & {ab['convergence_round']} & ${auc_a}$ & ${sen_a}$ & ${ab['dlg_ssim']}$ \\\\")
    t4 += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{ResNet-18 (44.0 MB Payload) Per-Round P2P Cryptographic Latency ($K=5$):}} \\",
        f"\\multicolumn{{3}}{{l}}{{X25519 ECDH + HKDF-SHA256 Handshake (RFC 7748)}} & \\multicolumn{{2}}{{r}}{{${rn['x25519_ecdh_key_agreement_ms']}\\text{{ ms}}$}} \\\\",
        f"\\multicolumn{{3}}{{l}}{{AES-256-CTR Encryption + Decryption (NIST SP 800-38A)}} & \\multicolumn{{2}}{{r}}{{${round(rn['aes_256_ctr_encrypt_44mb_ms'] + rn['aes_256_ctr_decrypt_44mb_ms'], 3)}\\text{{ ms}}$}} \\\\",
        f"\\multicolumn{{3}}{{l}}{{Ed25519 Sign + Verify (RFC 8032)}} & \\multicolumn{{2}}{{r}}{{${round(rn['ed25519_sign_44mb_ms'] + rn['ed25519_verify_44mb_ms'], 3)}\\text{{ ms}}$}} \\\\",
        f"\\multicolumn{{3}}{{l}}{{\\textbf{{Total Cryptographic Overhead per Round (44 MB)}}}} & \\multicolumn{{2}}{{r}}{{$\\mathbf{{{rn['total_per_round_crypto_overhead_ms']}\\text{{ ms}}$}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (t_dir / "table4_ablation_and_crypto.tex").write_text(nl.join(t4), encoding="utf-8")


def main():
    t0 = time.perf_counter()
    from src.evaluation.benchmarks import execute_full_project_pipeline
    master = execute_full_project_pipeline()
    export_latex_tables_from_json(master)

    d3_dir = ROOT_DIR / "results" / "day3"
    d3_dir.mkdir(parents=True, exist_ok=True)
    d3_report = {
        "status": "DAY_3_ALL_TASKS_COMPLETED_100_PERCENT_REAL",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "task_3_1_checkpoint_attacks": master["day1"]["checkpoint_attacks"],
        "task_3_1_final_fedmia": master["day1"]["fedmia_trajectory"][-1],
        "task_3_2_component_ablation": master["day3"]["exp4_component_ablation"],
        "task_3_2_resnet18_44mb_crypto": master["day3"]["exp4_resnet18_44mb_crypto_overhead"],
        "task_3_3_exported_latex_tables": [
            "results/tables/table1_privacy_utility.tex",
            "results/tables/table2_non_iid_dirichlet.tex",
            "results/tables/table3_inversion_checkpoints.tex",
            "results/tables/table4_ablation_and_crypto.tex",
        ],
    }
    (d3_dir / "day3_complete_metrics.json").write_text(json.dumps(d3_report, indent=2), encoding="utf-8")
    print("=== SPRINT DIA 3 COMPLETADO AL 100% ===")
    print(" -> 4 Tablas LaTeX exportadas en results/tables/")


if __name__ == "__main__":
    main()
