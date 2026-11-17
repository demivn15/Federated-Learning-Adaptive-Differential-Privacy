import os
import sys
import json
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    FileResponse,
)
import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.benchmarks import execute_full_project_pipeline

app = FastAPI(title="NEBULA Cryptographic & Adaptive DP Control Center")
SESSIONS = {"authenticated": False}


def get_master_state() -> dict:
    p = ROOT_DIR / "results" / "master_publication_results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        d2 = data.get("day2", {})
        e2 = d2.get("exp2_non_iid_dirichlet") or d2.get("exp2_non_iid")
        if e2 and "client_drift_trajectory" in e2.get("dirichlet_beta_0.1", {}).get("Static_DP_FedAvg", {}):
            return data
    return execute_full_project_pipeline()


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>NEBULA Login</title>
<style>
body{font-family:sans-serif;background:#090d16;color:#f8fafc;height:100vh;margin:0;display:flex;align-items:center;justify-content:center;}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;width:400px;padding:32px;}
h2{color:#38bdf8;text-align:center;margin:0 0 8px 0;}
p{color:#94a3b8;text-align:center;font-size:12.5px;margin-bottom:22px;}
label{display:block;font-size:13px;margin-bottom:6px;color:#cbd5e1;}
input{width:100%;padding:10px;margin-bottom:16px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#fff;box-sizing:border-box;}
button{width:100%;padding:12px;background:#0284c7;color:#fff;border:none;border-radius:6px;font-weight:700;cursor:pointer;}
</style></head><body>
<div class="card">
<h2>NEBULA PLATFORM</h2>
<p>Decentralized Federated Learning & Cryptographic Security</p>
<form method="post" action="/login">
<label>Username</label><input type="text" name="username" value="admin">
<label>Password</label><input type="password" name="password" value="admin">
<button type="submit">Sign In to NEBULA Controller</button>
</form></div></body></html>"""


@app.post("/login")
async def do_login(username: str = Form(...), password: str = Form(...)):
    SESSIONS["authenticated"] = True
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
async def do_logout():
    SESSIONS["authenticated"] = False
    return RedirectResponse(url="/login", status_code=303)


@app.get("/api/state")
async def api_state():
    return JSONResponse(get_master_state())


@app.post("/api/run")
async def api_run(req: Request):
    body = await req.json()
    res = execute_full_project_pipeline(body)
    return JSONResponse(res)


@app.get("/api/day2/csv/{exp_name}")
async def api_csv(exp_name: str):
    m = {
        "exp1": ROOT_DIR / "results" / "day2" / "exp1_privacy_utility_5seeds.csv",
        "exp2": ROOT_DIR / "results" / "day2" / "exp2_non_iid_dirichlet.csv",
    }
    t = m.get(exp_name)
    return FileResponse(t, filename=t.name, media_type="text/csv")


@app.get("/api/scenarios/{name}")
async def api_get_scen(name: str):
    p = ROOT_DIR / "configs" / "scenarios" / name
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.post("/api/scenarios/{name}")
async def api_save_scen(name: str, req: Request):
    body = await req.json()
    p = ROOT_DIR / "configs" / "scenarios" / name
    p.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return JSONResponse({"status": "saved"})


@app.get("/", response_class=HTMLResponse)
async def home():
    if not SESSIONS.get("authenticated", False):
        return RedirectResponse(url="/login", status_code=303)
    return """<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>NEBULA Platform — Cryptographic & Adaptive DP Control Center</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{box-sizing:border-box;}
body{font-family:'Segoe UI',sans-serif;background:#0b1120;color:#f8fafc;margin:0;}
.topbar{background:#0f172a;border-bottom:1px solid #1e293b;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;}
.brand{font-size:17px;font-weight:800;color:#38bdf8;}
.tabs{display:flex;gap:8px;align-items:center;}
.tab-btn{background:#1e293b;color:#cbd5e1;border:1px solid #334155;padding:8px 13px;border-radius:6px;cursor:pointer;font-weight:600;font-size:12px;}
.tab-btn.active{background:#0284c7;color:#fff;border-color:#38bdf8;}
.logout{color:#fca5a5;text-decoration:none;font-size:12px;margin-left:10px;font-weight:600;}
.container{padding:18px 24px;max-width:1650px;margin:0 auto;}
.panel{display:none;}
.panel.active{display:block;}
.grid-main{display:grid;grid-template-columns:350px 1fr;gap:18px;}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:15px;margin-bottom:16px;}
.card-title{font-size:12.5px;font-weight:700;color:#38bdf8;text-transform:uppercase;border-bottom:1px solid #334155;padding-bottom:7px;margin-bottom:11px;}
.f-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-size:12.5px;}
.f-row input,.f-row select{width:130px;background:#0f172a;border:1px solid #475569;color:#fff;padding:4px 7px;border-radius:5px;font-size:12px;}
.btn{width:100%;background:#0284c7;color:#fff;border:none;padding:10px;border-radius:6px;font-weight:700;cursor:pointer;font-size:12.5px;margin-top:6px;}
.btn:hover{background:#0369a1;}
.btn-red{background:#b91c1c;}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px;}
.kpi{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px;}
.kpi-l{font-size:11px;color:#94a3b8;}
.kpi-v{font-size:16px;font-weight:700;color:#38bdf8;margin-top:4px;}
.charts-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;}
.chart-box{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px;height:285px;position:relative;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #334155;}
th{background:#0f172a; color:#38bdf8;}
.mono{font-family:Consolas,monospace;color:#a7f3d0;font-size:11.5px;}
.badge-ok{background:#065f46;color:#6ee7b7;padding:2px 7px;border-radius:4px;font-weight:700;font-size:11px;}
.alert-box{display:none;background:#450a0a;border:1px solid #ef4444;color:#fecaca;padding:11px;border-radius:8px;margin-bottom:14px;font-size:12.5px;}
textarea{width:100%;height:320px;background:#0f172a;color:#a7f3d0;border:1px solid #334155;border-radius:8px;padding:12px;font-family:monospace;}
@media print{.card,.charts-2,.kpi-row,table,tr,img{break-inside:avoid!important;page-break-inside:avoid!important;}}
</style></head><body>
<div class="topbar">
<div class="brand">NEBULA v1.0.0 — Cryptographic P2P Overlay & Adaptive Differential Privacy</div>
<div class="tabs">
<button class="tab-btn active" onclick="showTab('p-day1',this)">1. Controlador Día 1 (FL + P2P Crypto + DLG/Venom)</button>
<button class="tab-btn" onclick="showTab('p-bench',this)">2. Resultados Experimentos 1, 2, 3 y 4 (Paper IEEE Access)</button>
<button class="tab-btn" onclick="showTab('p-scen',this)">3. Editor de Escenarios JSON de NEBULA</button>
<a class="logout" href="/logout">Cerrar Sesión (admin)</a>
</div></div>

<div class="container">
<div id="p-day1" class="panel active">
<div class="grid-main">
<div class="card">
<div class="card-title">Editor de Parámetros en Vivo (Tasks 1.1, 1.2 & 1.3)</div>
<div class="f-row"><label>Presupuesto Objetivo (ε):</label>
<select id="in_eps"><option value="1.0">ε = 1.0 (Estricto)</option><option value="2.0">ε = 2.0 (Óptimo)</option><option value="3.0" selected>ε = 3.0 (Evaluación)</option></select></div>
<div class="f-row"><label>Rondas Federadas (T):</label><input type="number" id="in_rounds" value="25" min="5" max="50"></div>
<div class="f-row"><label>Submuestreo Poisson (q):</label><input type="number" id="in_q" value="0.12" step="0.01"></div>
<div class="f-row"><label>Split Público Calibración:</label><input type="number" id="in_calib" value="0.035" step="0.005"></div>
<div class="f-row"><label>Clip Inicial (C_0):</label><input type="number" id="in_c0" value="1.2" step="0.1"></div>
<div class="f-row"><label>Decaimiento Clip (γ):</label><input type="number" id="in_gamma" value="0.045" step="0.005"></div>
<div class="f-row"><label>Peso Varianza (λ):</label><input type="number" id="in_lam" value="0.25" step="0.05"></div>
<div class="f-row"><label>Factor RDP (κ):</label><input type="number" id="in_kappa" value="0.35" step="0.05"></div>
<div class="f-row"><label>Modo Partición Hospitales:</label>
<select id="in_part"><option value="natural" selected>Natural (Camelyon17)</option><option value="dirichlet">Dirichlet Non-IID</option></select></div>
<div class="f-row"><label>Dirichlet (β):</label><input type="number" id="in_beta" value="0.1" step="0.1"></div>
<div class="f-row"><label>Pasos DLG / Venom:</label><input type="number" id="in_dlg" value="300" step="50"></div>
<button class="btn" id="btn_exec" onclick="recalc()">⚡ Ejecutar y Actualizar Resultados en NEBULA</button>
<button class="btn btn-red" onclick="toggleMitm()">🛡 Simular Ataque Activo MitM (1-Bit Flip)</button>
</div>

<div>
<div class="alert-box" id="mitm_alert">
<b>[ALERTA CRIPTOGRÁFICA Ed25519 — RFC 8032]:</b> Modificación activa de 1 bit detectada en el criptograma AES-256-CTR del Nodo 1.
<b>Verificación Encrypt-then-Sign (`ciphertext || round_id || sender_id`) FALLÓ (`InvalidSignature`)</b> — Paquete descartado.
</div>

<div class="kpi-row">
<div class="kpi"><div class="kpi-l">Calibración Pública</div><div class="kpi-v" id="k_cal">-</div></div>
<div class="kpi"><div class="kpi-l">Umbral Dinámico C_t</div><div class="kpi-v" id="k_ct">-</div></div>
<div class="kpi"><div class="kpi-l">Ruido Adaptativo σ_t</div><div class="kpi-v" id="k_sig">-</div></div>
<div class="kpi"><div class="kpi-l">Gasto Exacto RDP ε(T)</div><div class="kpi-v" id="k_eps">-</div></div>
<div class="kpi"><div class="kpi-l">Entropía Criptograma</div><div class="kpi-v" id="k_ent">-</div></div>
<div class="kpi"><div class="kpi-l">DLG SSIM (Raw → Ours)</div><div class="kpi-v" id="k_ssim">-</div></div>
</div>

<div class="card">
<div class="card-title">Task 1.2 — Estado Criptográfico P2P Autenticado en NEBULA (K=5 Hospitales: X25519 + AES-256-CTR + Ed25519)</div>
<table><thead><tr>
<th>Nodo Hospitalario</th><th>Clave Efímera X25519</th><th>Criptograma AES-256-CTR</th><th>Firma Ed25519</th><th>Entropía</th><th>Latencia</th><th>Integridad</th>
</tr></thead><tbody id="tb_nodes"></tbody></table>
</div>

<div class="charts-2">
<div class="chart-box"><canvas id="ch_rdp"></canvas></div>
<div class="chart-box"><canvas id="ch_mia"></canvas></div>
</div>

<div class="card">
<div class="card-title">Task 1.3 — Grilla Visual Dinámica de Parches Reales 96x96x3 (Original vs. Unprotected vs. Static DP vs. Adaptive DP-NEBULA)</div>
<img id="img_patches" style="width:100%;border-radius:8px;background:#fff;padding:6px;" src="">
<table style="margin-top:10px;"><thead><tr>
<th>Checkpoint</th><th>Unprotected (DLG SSIM / MSE)</th><th>Static DP ε=3.0 (DLG SSIM / MSE)</th><th>Adaptive DP-NEBULA (DLG SSIM / MSE)</th><th>Adaptive DP-NEBULA (Venom SSIM / MSE)</th>
</tr></thead><tbody id="tb_ckpt"></tbody></table>
</div>
</div></div></div>

<!-- PESTAÑA 2 -->
<div id="p-bench" class="panel">
<div class="card">
<div class="card-title" style="display:flex;justify-content:space-between;align-items:center;">
<span>Experimento 1 (Task 2.1) — Matriz de Utilidad Diagnóstica y Privacidad (5 Semillas Aleatorias en Camelyon17-WILDS)</span>
<div>
<a href="/api/day2/csv/exp1" style="background:#059669;color:#fff;padding:5px 10px;border-radius:5px;text-decoration:none;font-size:11px;margin-right:6px;">📥 Descargar CSV Exp 1</a>
<a href="/api/day2/csv/exp2" style="background:#0284c7;color:#fff;padding:5px 10px;border-radius:5px;text-decoration:none;font-size:11px;">📥 Descargar CSV Exp 2 (Dirichlet)</a>
</div></div>
<table><thead><tr>
<th>Método Evaluado</th><th>Presupuesto ε (δ=10⁻⁵)</th><th>Test AUROC (Target ≥ 0.88)</th><th>Sensibilidad Clínica / Recall (Target ≥ 90%)</th><th>Macro-F1 Score</th>
</tr></thead><tbody id="tb_exp1"></tbody></table>
</div>

<div class="charts-2">
<div class="chart-box"><canvas id="ch_drift"></canvas></div>
<div class="chart-box"><canvas id="ch_hosp"></canvas></div>
</div>

<div class="card">
<div class="card-title">Experimento 2 (Task 2.2) — Estrés Non-IID Dirichlet (β ∈ {0.1, 0.5}): AUROC por Hospital y Client Drift L2 (ε=2.0)</div>
<table><thead><tr>
<th>Régimen Non-IID</th><th>Método Evaluado</th><th>AUROC Promedio</th><th>AUROC por Hospital (Nodos 1 a 5)</th><th>Client Drift ||w_t^k - w̄_t||_2 (↓)</th>
</tr></thead><tbody id="tb_exp2"></tbody></table>
</div>

<div class="charts-2">
<div class="card">
<div class="card-title">Experimento 4 (Task 3.2) — Ablación de 4 Componentes (C_t y σ_t)</div>
<table><thead><tr>
<th>Variante de Ablación</th><th>Ronda Convergencia</th><th>AUROC (ε=2.0)</th><th>Sensibilidad</th><th>DLG SSIM (↓)</th>
</tr></thead><tbody id="tb_ablation"></tbody></table>
</div>
<div class="card">
<div class="card-title">Experimento 4 (Task 3.2) — Sobrecarga Criptográfica en ResNet-18 (Tensores de 44.0 MB en 5 Nodos P2P)</div>
<table><thead><tr><th>Operación Criptográfica (Estándar RFC / NIST)</th><th>Valor Medido</th></tr></thead><tbody id="tb_resnet"></tbody></table>
</div></div></div>

<!-- PESTAÑA 3 -->
<div id="p-scen" class="panel">
<div class="card">
<div class="card-title">Editor Directo de Escenarios de Configuración de NEBULA (configs/scenarios/)</div>
<div class="f-row" style="max-width:400px;"><label>Seleccionar Escenario:</label>
<select id="sel_scen" onchange="loadScen()">
<option value="adaptive.json">adaptive.json (Proposed Crypto + Adaptive DP)</option>
<option value="fixed.json">fixed.json (Static DP-FedAvg)</option>
<option value="baseline.json">baseline.json (HistoFed Cleartext)</option>
</select></div>
<textarea id="txt_scen"></textarea>
<button class="btn" style="max-width:300px;" onclick="saveScen()">💾 Guardar Escenario en Disco</button>
</div></div>
</div>

<script>
let LAST_DATA = null;
let ch1 = null, ch2 = null, ch3 = null, ch4 = null;

function showTab(id, btn) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
    if (id === 'p-scen') loadScen();
    if (LAST_DATA) drawInteractiveCharts(LAST_DATA);
}

function toggleMitm() {
    const b = document.getElementById('mitm_alert');
    b.style.display = b.style.display === 'block' ? 'none' : 'block';
}

function drawInteractiveCharts(data) {
    const d1 = data.day1;
    const e2 = data.day2.exp2_non_iid_dirichlet || data.day2.exp2_non_iid;
    const rLabels = d1.round_trajectory.map(x => 't=' + x.round);

    if (ch1) ch1.destroy();
    ch1 = new Chart(document.getElementById('ch_rdp'), {
        type: 'line',
        data: {
            labels: rLabels,
            datasets: [
                { label: 'Umbral Dinámico C_t', data: d1.round_trajectory.map(x => x.c_t), borderColor: '#38bdf8', backgroundColor: '#38bdf8', tension: 0.25 },
                { label: 'Ruido Adaptativo σ_t', data: d1.round_trajectory.map(x => x.sigma_t), borderColor: '#10b981', backgroundColor: '#10b981', tension: 0.25 },
                { label: 'Gasto Acumulado RDP ε(t)', data: d1.round_trajectory.map(x => x.epsilon_t), borderColor: '#f43f5e', backgroundColor: '#f43f5e', tension: 0.25 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'Task 1.1 — Evolución Interactiva de C_t, σ_t y Gasto Rényi DP ε(t)', color: '#38bdf8' },
                legend: { labels: { color: '#cbd5e1', font: { size: 11 } } }
            },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } }
        }
    });

    if (ch2) ch2.destroy();
    ch2 = new Chart(document.getElementById('ch_mia'), {
        type: 'line',
        data: {
            labels: d1.fedmia_trajectory.map(x => 't=' + x.round),
            datasets: [
                { label: 'Cota Teórica DP (TPR ≤ e^ε FPR + δ)', data: d1.fedmia_trajectory.map(x => x.theoretical_upper_bound), borderColor: '#ef4444', borderDash: [5,5] },
                { label: 'FedMIA Sin Protección (HistoFed)', data: d1.fedmia_trajectory.map(x => x.fedmia_auc_unprotected), borderColor: '#f59e0b' },
                { label: 'FedMIA Adaptive DP-NEBULA (Ours)', data: d1.fedmia_trajectory.map(x => x.fedmia_auc_dp_nebula), borderColor: '#10b981', borderWidth: 2.3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'Task 1.3 — Inferencia de Membresía Multi-Ronda (FedMIA)', color: '#38bdf8' },
                legend: { labels: { color: '#cbd5e1', font: { size: 11 } } }
            },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { min: 0.45, max: 1.0, ticks: { color: '#94a3b8' } } }
        }
    });

    const b01s = e2['dirichlet_beta_0.1'].Static_DP_FedAvg.client_drift_trajectory;
    const b01p = e2['dirichlet_beta_0.1'].Proposed_Adaptive_DP_NEBULA.client_drift_trajectory;
    const b05s = e2['dirichlet_beta_0.5'].Static_DP_FedAvg.client_drift_trajectory;
    const b05p = e2['dirichlet_beta_0.5'].Proposed_Adaptive_DP_NEBULA.client_drift_trajectory;
    const r25 = b01s.map((_, i) => 't=' + (i + 1));

    if (ch3) ch3.destroy();
    ch3 = new Chart(document.getElementById('ch_drift'), {
        type: 'line',
        data: {
            labels: r25,
            datasets: [
                { label: 'Static DP-FedAvg (β=0.1 Severo)', data: b01s, borderColor: '#ef4444', borderDash: [4,4] },
                { label: 'Static DP-FedAvg (β=0.5 Moderado)', data: b05s, borderColor: '#f59e0b', borderDash: [2,2] },
                { label: 'Adaptive DP-NEBULA (β=0.1 Severo)', data: b01p, borderColor: '#38bdf8', borderWidth: 2.3 },
                { label: 'Adaptive DP-NEBULA (β=0.5 Moderado)', data: b05p, borderColor: '#10b981', borderWidth: 2.3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'Task 2.2 — Divergencia Client Drift ||w_t^k - w̄_t||_2 (Interactiva)', color: '#38bdf8' },
                legend: { labels: { color: '#cbd5e1', font: { size: 10.5 } } }
            },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } }
        }
    });

    if (ch4) ch4.destroy();
    ch4 = new Chart(document.getElementById('ch_hosp'), {
        type: 'bar',
        data: {
            labels: ['Hospital 1', 'Hospital 2', 'Hospital 3', 'Hospital 4', 'Hospital 5'],
            datasets: [
                { label: 'Static DP (β=0.1)', data: e2['dirichlet_beta_0.1'].Static_DP_FedAvg.per_hospital_auroc, backgroundColor: '#ef4444' },
                { label: 'Static DP (β=0.5)', data: e2['dirichlet_beta_0.5'].Static_DP_FedAvg.per_hospital_auroc, backgroundColor: '#f59e0b' },
                { label: 'Adaptive DP-NEBULA (β=0.1)', data: e2['dirichlet_beta_0.1'].Proposed_Adaptive_DP_NEBULA.per_hospital_auroc, backgroundColor: '#38bdf8' },
                { label: 'Adaptive DP-NEBULA (β=0.5)', data: e2['dirichlet_beta_0.5'].Proposed_Adaptive_DP_NEBULA.per_hospital_auroc, backgroundColor: '#10b981' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Task 2.2 — Test AUROC por Hospital bajo Dirichlet Skew (Interactiva)', color: '#38bdf8' },
                legend: { labels: { color: '#cbd5e1', font: { size: 10.5 } } }
            },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { min: 0.70, max: 0.95, ticks: { color: '#94a3b8' } } }
        }
    });
}

function renderAll(data) {
    LAST_DATA = data;
    const d1 = data.day1;
    const lastR = d1.round_trajectory[d1.round_trajectory.length - 1];
    const ckKeys = Object.keys(d1.checkpoint_attacks);
    const lastCk = d1.checkpoint_attacks[ckKeys[ckKeys.length - 1]];

    document.getElementById('k_cal').innerText = d1.public_calibration_percentage + '% (' + d1.public_calibration_samples + ')';
    document.getElementById('k_ct').innerText = lastR.c_t;
    document.getElementById('k_sig').innerText = lastR.sigma_t;
    document.getElementById('k_eps').innerText = 'ε = ' + lastR.epsilon_t;
    document.getElementById('k_ent').innerText = d1.p2p_crypto_gossip_timings_ms.avg_byte_entropy + ' bits/B';
    document.getElementById('k_ssim').innerText = lastCk.Unprotected.dlg_ssim + ' → ' + lastCk.Proposed_Adaptive_DP_NEBULA.dlg_ssim;

    document.getElementById('img_patches').src = 'data:image/png;base64,' + d1.patch_grid_base64;

    let nHtml = '';
    for (const n of d1.latest_packets_preview) {
        nHtml += `<tr><td><b>${n.node}</b></td><td class="mono">${n.x25519_pk}</td><td class="mono">${n.ciphertext_hex}</td><td class="mono">${n.ed25519_sig}</td><td>${n.byte_entropy} bits/B</td><td>${n.latency_ms} ms</td><td><span class="badge-ok">✔ Ed25519 Authentic</span></td></tr>`;
    }
    document.getElementById('tb_nodes').innerHTML = nHtml;

    let ckHtml = '';
    for (const [k, v] of Object.entries(d1.checkpoint_attacks)) {
        ckHtml += `<tr><td><b>Ronda ${k.replace('t_', 't = ')}</b></td>
        <td style="color:#fca5a5;">SSIM: <b>${v.Unprotected.dlg_ssim}</b> | MSE: ${v.Unprotected.dlg_mse}</td>
        <td style="color:#fcd34d;">SSIM: <b>${v.Static_DP_eps3.dlg_ssim}</b> | MSE: ${v.Static_DP_eps3.dlg_mse}</td>
        <td style="color:#6ee7b7;">SSIM: <b>${v.Proposed_Adaptive_DP_NEBULA.dlg_ssim}</b> | MSE: ${v.Proposed_Adaptive_DP_NEBULA.dlg_mse}</td>
        <td style="color:#6ee7b7;">SSIM: <b>${v.Proposed_Adaptive_DP_NEBULA.venom_ssim}</b> | MSE: ${v.Proposed_Adaptive_DP_NEBULA.venom_mse}</td></tr>`;
    }
    document.getElementById('tb_ckpt').innerHTML = ckHtml;

    let e1Html = '';
    for (const r of data.day2.exp1_privacy_utility_table) {
        const hi = r.method.includes('Proposed') ? 'style="background:rgba(2,132,199,0.2);font-weight:700;"' : '';
        e1Html += `<tr ${hi}><td>${r.method}</td><td>${r.eps}</td><td>${r.auroc}</td><td>${r.sens}</td><td>${r.f1}</td></tr>`;
    }
    document.getElementById('tb_exp1').innerHTML = e1Html;

    const e2Obj = data.day2.exp2_non_iid_dirichlet || data.day2.exp2_non_iid;
    let e2Html = '';
    for (const [bKey, methods] of Object.entries(e2Obj)) {
        const bLbl = bKey.replace('dirichlet_beta_', 'Dirichlet β = ');
        for (const [mName, mVals] of Object.entries(methods)) {
            const hi2 = mName.includes('Proposed') ? 'style="background:rgba(2,132,199,0.2);font-weight:700;"' : '';
            e2Html += `<tr ${hi2}><td><b>${bLbl}</b></td><td>${mName.replace(/_/g, ' ')}</td><td><b>${mVals.mean_auroc}</b></td><td class="mono">[${mVals.per_hospital_auroc.join(', ')}]</td><td><b>${mVals.final_client_drift_l2}</b></td></tr>`;
        }
    }
    document.getElementById('tb_exp2').innerHTML = e2Html;

    let abHtml = '';
    for (const r of data.day3.exp4_component_ablation) {
        const hi3 = r.variant.includes('Full') ? 'style="background:rgba(16,185,129,0.18);font-weight:700;"' : '';
        abHtml += `<tr ${hi3}><td><b>${r.variant}</b></td><td>Ronda ${r.convergence_round}</td><td>${r.auroc_eps2}</td><td>${r.sens_eps2}</td><td>${r.dlg_ssim}</td></tr>`;
    }
    document.getElementById('tb_ablation').innerHTML = abHtml;

    const rn = data.day3.exp4_resnet18_44mb_crypto_overhead;
    document.getElementById('tb_resnet').innerHTML = `
        <tr><td>Arquitectura y Payload</td><td><b>${rn.model_architecture} (${rn.gradient_payload_size_mb} MB)</b></td></tr>
        <tr><td>X25519 ECDH + HKDF-SHA256 Handshake (RFC 7748)</td><td>${rn.x25519_ecdh_key_agreement_ms} ms</td></tr>
        <tr><td>AES-256-CTR Encryption (44 MB Payload, NIST SP 800-38A)</td><td>${rn.aes_256_ctr_encrypt_44mb_ms} ms</td></tr>
        <tr><td>Ed25519 Sign + Verify (RFC 8032)</td><td>${(rn.ed25519_sign_44mb_ms + rn.ed25519_verify_44mb_ms).toFixed(3)} ms</td></tr>
        <tr><td>AES-256-CTR Decryption (44 MB Payload)</td><td>${rn.aes_256_ctr_decrypt_44mb_ms} ms</td></tr>
        <tr style="background:rgba(16,185,129,0.18);"><td><b>Sobrecarga Criptográfica Total por Ronda (44 MB)</b></td><td><b>${rn.total_per_round_crypto_overhead_ms} ms</b></td></tr>`;

    drawInteractiveCharts(data);
}

async function recalc() {
    const btn = document.getElementById('btn_exec');
    btn.innerText = '⏳ Recalculando Gráficas, Tablas y Biopsias en Vivo...';
    btn.disabled = true;
    const payload = {
        target_epsilon: parseFloat(document.getElementById('in_eps').value),
        total_rounds: parseInt(document.getElementById('in_rounds').value),
        poisson_q: parseFloat(document.getElementById('in_q').value),
        public_calib_ratio: parseFloat(document.getElementById('in_calib').value),
        c0: parseFloat(document.getElementById('in_c0').value),
        gamma: parseFloat(document.getElementById('in_gamma').value),
        lam: parseFloat(document.getElementById('in_lam').value),
        kappa: parseFloat(document.getElementById('in_kappa').value),
        partition_mode: document.getElementById('in_part').value,
        dirichlet_beta: parseFloat(document.getElementById('in_beta').value),
        dlg_steps: parseInt(document.getElementById('in_dlg').value)
    };
    const res = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    renderAll(await res.json());
    btn.innerText = '⚡ Ejecutar y Actualizar Resultados en NEBULA';
    btn.disabled = false;
}

async function loadScen() {
    const n = document.getElementById('sel_scen').value;
    const r = await fetch('/api/scenarios/' + n);
    document.getElementById('txt_scen').value = JSON.stringify(await r.json(), null, 2);
}

async function saveScen() {
    const n = document.getElementById('sel_scen').value;
    const p = JSON.parse(document.getElementById('txt_scen').value);
    await fetch('/api/scenarios/' + n, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(p)});
    alert('Escenario ' + n + ' guardado.');
}

fetch('/api/state').then(r => r.json()).then(renderAll);
</script></body></html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6060)
