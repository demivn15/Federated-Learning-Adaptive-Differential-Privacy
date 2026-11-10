// Behavioral checks for controls, historic labels, comparisons and CSV evidence.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.join(here,'nebula/nebula/frontend');
const names = ['smoke_adaptive','smoke_fixed','smoke_baseline'];
// Generated from measured journals by verify_adaptive_release.py.
const measured = JSON.parse(await fs.readFile(path.join(here,'test_artifacts/frontend_evidence_v3.json'),'utf8'));
const data = new Map(measured.map(d=>[d.scenario.name,d]));
function elements() {
    const nodes = new Map();
    return {nodes, document:{
        getElementById(id) {
            if(!nodes.has(id)) nodes.set(id,{dataset:{scenario:'smoke_adaptive'},value:'',checked:false,
                style:{},innerHTML:'',textContent:'',appendChild(){},addEventListener(){}});
            return nodes.get(id);
        },
        createElement(){return {click(){}};},
    }};
}

const controls = elements();
const controlContext = vm.createContext({document:controls.document,window:{}});
const controlModule = new vm.SourceTextModule(await fs.readFile(path.join(frontend,'static/js/deployment/dp.js'),'utf8'),{context:controlContext});
await controlModule.link(()=>{});
await controlModule.evaluate();
const manager = controlModule.namespace.default;
manager.initializeDifferentialPrivacy();
manager.setDpConfig({enabled:true,mode:'adaptive'});
assert.equal(controls.nodes.get('dp-adaptive-settings').style.display,'block');
assert.equal(manager.getDpConfig().adaptive_quantile,.5);
assert.throws(()=>manager.setDpConfig({mode:'dynamic'}),/programado anterior/);
controls.nodes.get('dpAdaptiveQuantile').value='1';
assert.throws(()=>manager.getDpConfig(),/Cuantil/);
manager.setDpConfig({enabled:true,mode:'fixed'});
assert.equal(controls.nodes.get('dp-adaptive-settings').style.display,'none');

const view = elements();
const blobs = [];
const context = vm.createContext({document:view.document,Blob,
    URL:{createObjectURL(blob){blobs.push(blob);return 'blob:test';},revokeObjectURL(){}},
    setTimeout(){},setInterval(){},
    fetch:async url => ({ok:true,status:200,json:async()=>url==='/platform/api/dashboard'
        ? measured.map(d=>d.scenario) : data.get(decodeURIComponent(url).split('/').at(-2))}),
});
const source = await fs.readFile(path.join(frontend,'static/js/research.js'),'utf8');
const module = new vm.SourceTextModule(source+'\nexport {render,mode};',{context});
await module.link(()=>{});
await module.evaluate();
assert.match(view.nodes.get('p2-privacy').textContent,/DP adaptativa/);
assert.match(view.nodes.get('p2-adaptation').innerHTML,/Cuantil observado/);
assert.equal((view.nodes.get('p2-adaptation').innerHTML.match(/<tr>/g)||[]).length,10);
assert.match(view.nodes.get('p2-clip-chart').innerHTML,/<polyline/);
view.nodes.get('p2-csv').onclick();
const csv = (await blobs.at(-1).text()).trim().split(/\r?\n/).map(line=>line.split(','));
assert.equal(csv.length,10);
const header = csv[0];
for(const row of csv.slice(1)) {
    assert.equal(row.length,header.length);
    assert.equal(row[header.indexOf('mode')],'adaptive');
    assert.equal(Number(row[header.indexOf('adaptive_reference_records')]),91);
    assert.ok(Math.abs(Number(row[header.indexOf('noise_std_sum')])-Number(row[header.indexOf('sigma')])*Number(row[header.indexOf('clip')]))<1e-12);
}
view.nodes.get('p2-compare-select').value='smoke_fixed';
await view.nodes.get('p2-add').onclick();
assert.match(view.nodes.get('p2-comparison').innerHTML,/DP fija/);
assert.match(view.nodes.get('p2-comparison').innerHTML,/misma configuración base/);

const historic = structuredClone(measured[0]);
historic.protocol='project2-public-evaluation-v2';
historic.comparison_key.protocol=historic.protocol;
historic.dp.mode='dynamic';
historic.scenario.name='historic_dynamic';
historic.privacy.forEach(r=>delete r.adaptation);
assert.equal(module.namespace.mode(historic),'DP programada (v2)');
module.namespace.render(historic);
assert.match(view.nodes.get('p2-explanation').textContent,/no fue adaptativa/);
view.nodes.get('p2-compare-select').value='smoke_fixed';
await view.nodes.get('p2-add').onclick();
assert.match(view.nodes.get('p2-comparison').innerHTML,/protocolo\/configuración diferente/);
console.log('PASS: adaptive controls, legacy labels, real measured CSV rows, clipping charts and protocol comparisons.');
