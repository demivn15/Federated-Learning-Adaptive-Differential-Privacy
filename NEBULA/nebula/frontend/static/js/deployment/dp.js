// Fixed sigma, with fixed or public-reference adaptive clipping.
const DpManager = (() => {
    const defaults = {enabled:false, mode:'fixed', target_epsilon:8, target_delta:0.00001,
        noise_multiplier:1, noise_end:0.7, max_grad_norm:1, accountant:'rdp',
        secure_mode:false, poisson_sampling:true, clipping:'flat',
        adaptive_quantile:0.5, adaptive_smoothing:0.5, adaptive_min_clip:0.1,
        adaptive_max_clip:5, adaptive_max_records:91};
    const fields = {mode:'dpMode', target_epsilon:'dpEpsilon', target_delta:'dpDelta',
        max_grad_norm:'dpMaxGradNorm', adaptive_quantile:'dpAdaptiveQuantile',
        adaptive_smoothing:'dpAdaptiveSmoothing', adaptive_min_clip:'dpAdaptiveMinClip',
        adaptive_max_clip:'dpAdaptiveMaxClip'};
    function toggle() {
        const enabled = document.getElementById('dpSwitch').checked;
        document.getElementById('dp-settings').style.display = enabled ? 'block' : 'none';
        document.getElementById('dp-adaptive-settings').style.display = document.getElementById('dpMode').value === 'adaptive' ? 'block' : 'none';
    }
    function setDpConfig(config={}) {
        if (config.mode === 'dynamic') throw new Error('Esta configuración corresponde al ruido programado anterior. Selecciona explícitamente DP fija o adaptativa para un escenario nuevo.');
        const c = {...defaults,...config};
        if (!['fixed','adaptive'].includes(c.mode)) throw new Error('Modo DP no válido.');
        document.getElementById('dpSwitch').checked = c.enabled;
        document.getElementById('dpSecureMode').checked = c.secure_mode;
        for (const [key,id] of Object.entries(fields)) document.getElementById(id).value = c[key];
        toggle();
    }
    function getDpConfig() {
        const c = {...defaults, enabled:document.getElementById('dpSwitch').checked,
            secure_mode:document.getElementById('dpSecureMode').checked};
        for (const [key,id] of Object.entries(fields)) {
            c[key] = key === 'mode' ? document.getElementById(id).value : Number(document.getElementById(id).value);
            if (key !== 'mode' && (!Number.isFinite(c[key]) || c[key] <= 0)) throw new Error('Los parámetros DP deben ser números positivos.');
        }
        if (c.target_delta >= 1) throw new Error('Delta debe estar entre 0 y 1.');
        if (!['fixed','adaptive'].includes(c.mode)) throw new Error('Selecciona DP fija o adaptativa.');
        if (c.adaptive_quantile >= 1 || c.adaptive_smoothing > 1) throw new Error('Cuantil: entre 0 y 1. Peso de adaptación: mayor que 0 y hasta 1.');
        if (c.adaptive_min_clip > c.adaptive_max_clip) throw new Error('C mínimo no puede superar C máximo.');
        if (c.mode === 'adaptive' && (c.max_grad_norm < c.adaptive_min_clip || c.max_grad_norm > c.adaptive_max_clip)) throw new Error('C inicial debe estar entre C mínimo y C máximo.');
        return c;
    }
    function initializeDifferentialPrivacy() {
        document.getElementById('dpSwitch').addEventListener('change', () => {
            if (document.getElementById('dpSwitch').checked && window.AdversarialTrainingManager)
                window.AdversarialTrainingManager.setAdversarialTrainingConfig({enabled:false});
            toggle();
        });
        document.getElementById('dpMode').addEventListener('change',toggle);
        setDpConfig();
    }
    return {initializeDifferentialPrivacy,getDpConfig,setDpConfig,resetDpConfig:()=>setDpConfig()};
})();
export default DpManager;
