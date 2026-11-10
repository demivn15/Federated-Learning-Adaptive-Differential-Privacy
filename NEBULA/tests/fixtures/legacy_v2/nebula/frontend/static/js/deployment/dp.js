// Project 2 controls. Epsilon is a ceiling; the noise is calibrated before training.
const DpManager = (() => {
    const defaults = {enabled:false, mode:'fixed', target_epsilon:8, target_delta:0.00001,
        noise_multiplier:1, noise_end:0.7, max_grad_norm:1, accountant:'rdp',
        secure_mode:false, poisson_sampling:true, clipping:'flat'};
    const fields = {mode:'dpMode', target_epsilon:'dpEpsilon', target_delta:'dpDelta',
        noise_multiplier:'dpNoiseMultiplier', noise_end:'dpNoiseEnd', max_grad_norm:'dpMaxGradNorm'};
    function toggle() {
        const enabled = document.getElementById('dpSwitch').checked;
        document.getElementById('dp-settings').style.display = enabled ? 'block' : 'none';
        document.getElementById('dp-dynamic-settings').style.display = document.getElementById('dpMode').value === 'dynamic' ? 'block' : 'none';
    }
    function setDpConfig(config={}) {
        const c = {...defaults,...config};
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
