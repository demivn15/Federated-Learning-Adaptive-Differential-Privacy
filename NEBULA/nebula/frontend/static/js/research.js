const app = document.getElementById('research-app');
const scenario = app.dataset.scenario;
const el = id => document.getElementById(id);
let current = null;
const compared = new Map();
const palette = ['#71d6d1','#f5bb75','#aaabff'];
const escape = value => String(value ?? '—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value,digits=3) => value == null ? 'pendiente' : Number(value).toFixed(digits);
const percent = value => value == null ? 'pendiente' : `${(value*100).toFixed(1)} %`;
const supported = d => ['project2-public-evaluation-v2','project2-public-evaluation-v3'].includes(d.protocol);
const mode = d => !d.dp.enabled ? 'Sin DP' : d.dp.mode === 'adaptive' ? 'DP adaptativa' : d.dp.mode === 'dynamic' ? 'DP programada (v2)' : 'DP fija';
const table = (headers,rows) => `<table><thead><tr>${headers.map(h=>`<th>${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${escape(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
function chart(target,series,limit=null) {
    const points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[1]));
    if (!points.length) { el(target).textContent='Sin datos todavía.';return; }
    const xmax=Math.max(1,...points.map(p=>p[0]));
    const ymax=Math.max(0.01,limit??0,...points.map(p=>p[1]))*1.1;
    const x=r=>45+455*(r-1)/Math.max(1,xmax-1), y=v=>172-145*v/ymax;
    let svg='<svg viewBox="0 0 540 225" role="img" aria-label="Evolución por ronda">';
    for(let i=0;i<4;i++){const v=i*ymax/3;svg+=`<line x1="45" x2="500" y1="${y(v)}" y2="${y(v)}" stroke="#33465f"/><text x="2" y="${y(v)+4}">${v.toFixed(2)}</text>`;}
    for(let r=1;r<=xmax;r++)svg+=`<text x="${x(r)}" y="193" text-anchor="middle">${r}</text>`;
    if(limit!=null)svg+=`<line x1="45" x2="500" y1="${y(limit)}" y2="${y(limit)}" stroke="#ffbf89" stroke-dasharray="5 4"/>`;
    series.forEach((s,i)=>{svg+=`<polyline fill="none" stroke="${palette[i%3]}" stroke-width="2.5" points="${s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}"/>`;s.points.forEach(p=>{svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3.5" fill="${palette[i%3]}"><title>${escape(s.name)} · ronda ${p[0]}: ${number(p[1])}</title></circle>`;});});
    el(target).innerHTML=svg+'<text x="273" y="216" text-anchor="middle">Ronda</text></svg><div class="p2-legend">'+series.map((s,i)=>`<span style="color:${palette[i%3]}">${escape(s.name)}</span>`).join('')+'</div>';
}
function render(d) {
    current=d;
    const legacy=!supported(d);
    el('p2-status').textContent=legacy ? 'Ejecución anterior al protocolo v2. No contiene estos registros de privacidad/evaluación.' : `${d.execution} · ${d.scenario.title??scenario} · estado: ${d.scenario.status} · ${d.errors.length ? 'hay errores de ejecución que revisar' : 'evidencia recibida'}`;
    el('p2-federation').textContent=`1 servidor + ${d.expected_clients.length} clientes · ${d.rounds} rondas`;
    el('p2-privacy').textContent=d.dp.enabled ? `${mode(d)} · ε máximo observado: ${number(d.epsilon_max)} / ${number(d.dp.target_epsilon)}` : 'Sin DP: no existe una cota de privacidad';
    el('p2-evaluation').textContent=d.final_test ? `Test final disponible · ${d.final_test.records} registros` : `Pendiente · ${d.evaluations.length} rondas con validación pública`;
    el('p2-explanation').textContent=!d.dp.enabled ? 'Esta ejecución sirve como referencia de aprendizaje federado sin protección diferencial.' : d.dp.mode==='adaptive' ? `El límite C se adapta al cuantil de gradientes del modelo actual en la referencia pública. σ permanece constante y la escala de ruido es σ × C. El contador acumula ε con objetivo ${d.dp.target_epsilon} y δ=${d.dp.target_delta}. No se consultan los gradientes privados ni el test final.` : d.dp.mode==='dynamic' ? `Evidencia histórica v2: el ruido seguía un calendario definido antes del entrenamiento. Esta ejecución no fue adaptativa y conserva sus resultados originales.` : `El multiplicador σ y el límite C permanecen constantes. σ se calibra para el presupuesto ε=${d.dp.target_epsilon}, δ=${d.dp.target_delta}.`;
    el('p2-rng').textContent=d.dp.enabled ? `${d.rng}. ${d.accounting_complete ? 'Contabilidad de las rondas terminada.' : 'Contabilidad aún incompleta.'} ${d.errors.map(e=>e.error).join(' ')}` : '';
    const m=d.final_test;
    el('p2-metrics').innerHTML=[['Exactitud',m?.accuracy,'Clasificaciones correctas sobre todo el test.'],['Recall maligno',m?.malignant_recall,'Casos malignos detectados entre los malignos reales.'],['F1 maligno',m?.malignant_f1,'Equilibrio entre precisión y recall de malignos.'],['ROC-AUC',m?.roc_auc,'Separación entre clases según la puntuación.']].map(([title,v,desc])=>`<article><h2>${title}</h2><strong class="metric">${title==='ROC-AUC'?number(v):percent(v)}</strong><small>${desc}</small></article>`).join('');
    const latest=new Map();d.privacy.forEach(r=>latest.set(r.participant,r));
    const ids=d.expected_clients;
    chart('p2-epsilon-chart',ids.map(i=>({name:`Cliente ${i}`,points:d.privacy.filter(r=>r.participant===i).map(r=>[r.round,r.epsilon])})),d.dp.enabled?d.dp.target_epsilon:null);
    chart('p2-noise-chart',ids.map(i=>({name:`Cliente ${i}`,points:d.privacy.filter(r=>r.participant===i).map(r=>[r.round,r.sigma])})));
    chart('p2-clip-chart',ids.map(i=>({name:`Cliente ${i}`,points:d.privacy.filter(r=>r.participant===i).map(r=>[r.round,r.clip])})));
    chart('p2-noise-std-chart',ids.map(i=>({name:`Cliente ${i}`,points:d.privacy.filter(r=>r.participant===i).map(r=>[r.round,r.noise_std_sum??r.sigma*r.clip])})));
    const feedback=d.privacy.filter(r=>r.adaptation);
    el('p2-adaptation').innerHTML=feedback.length ? table(['Cliente','Ronda','Referencia pública','C anterior','Cuantil observado','Objetivo limitado','C aplicado','σ × C'],feedback.map(r=>[r.participant,r.round,r.adaptation.reference_records,number(r.adaptation.previous_clip),number(r.adaptation.observed_quantile),number(r.adaptation.target_clip),number(r.clip),number(r.noise_std_sum)])) : d.dp.mode==='adaptive' ? 'Pendiente de la primera ronda adaptativa.' : 'Este modo no toma decisiones adaptativas.';
    chart('p2-quality-chart',[{name:'Exactitud pública (%)',points:d.evaluations.map(r=>[r.round,100*r.validation.accuracy])},{name:'Recall maligno público (%)',points:d.evaluations.map(r=>[r.round,100*r.validation.malignant_recall])}],100);
    el('p2-distribution').innerHTML=d.manifest ? table(['Cliente','Registros','Malignos','Benignos'],d.manifest.clients.map(c=>[c.participant,c.records,c.malignant,c.benign])) : 'No disponible para este protocolo.';
    el('p2-clients').innerHTML=table(['Cliente','Ronda','ε acumulado','Límite ε','σ','C','q','Pasos acumulados'],ids.map(i=>{const r=latest.get(i);return [i,r?.round??'pendiente',number(r?.epsilon),d.dp.enabled?number(d.dp.target_epsilon):'sin cota',number(r?.sigma),number(r?.clip),number(r?.sample_rate),r?.cumulative_steps??'—'];}));
    el('p2-confusion').innerHTML=m ? table(['Real / predicción','Maligno','Benigno'],[['Maligno',...m.confusion_matrix[0]],['Benigno',...m.confusion_matrix[1]]])+`<p>Malignos clasificados como benignos: <strong>${escape(m.confusion_matrix[0][1])}</strong>.</p>` : 'Disponible cuando termina el test final.';
    el('p2-updated').textContent=`Última lectura: ${new Date().toLocaleTimeString()}`;
    compared.set(scenario,d);renderComparison();
}
function renderComparison(){
    el('p2-comparison').innerHTML=table(['Escenario','Modo','Exactitud','Recall maligno','F1 maligno','ε final','Suma tiempo rondas','Comparabilidad'],[...compared.values()].map(d=>{
        const same=supported(d) && JSON.stringify(d.comparison_key)===JSON.stringify(current.comparison_key);
        const budgetSame=!d.dp.enabled||!current.dp.enabled||['target_epsilon','target_delta','max_grad_norm'].every(k=>d.dp[k]===current.dp[k]);
        const adaptiveSame=d.dp.mode!=='adaptive'||current.dp.mode!=='adaptive'||['adaptive_quantile','adaptive_smoothing','adaptive_min_clip','adaptive_max_clip','adaptive_max_records'].every(k=>d.dp[k]===current.dp[k]);
        return [d.scenario.title??d.scenario.name,mode(d),percent(d.final_test?.accuracy),percent(d.final_test?.malignant_recall),percent(d.final_test?.malignant_f1),d.dp.enabled?number(d.epsilon_max):'sin cota',d.evaluations.length ? `${number(d.evaluations.reduce((s,r)=>s+r.round_seconds,0),1)} s` : "pendiente",!same?'protocolo/configuración diferente':!budgetSame?'presupuesto DP/C inicial diferente':!adaptiveSame?'configuración adaptativa diferente':!d.final_test?'pendiente':'misma configuración base'];
    }));
}
async function refresh(){
    try{const r=await fetch(`/platform/api/dashboard/${encodeURIComponent(scenario)}/research`);if(!r.ok)throw Error(`HTTP ${r.status}`);render(await r.json());}
    catch(e){el('p2-status').textContent=`No se pudo actualizar: ${e.message}. Los datos anteriores permanecen visibles.`;}
}
function download(data,type,filename){const url=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
el('p2-json').onclick=()=>current&&download(JSON.stringify(current,null,2),'application/json',`${scenario}_evidence.json`);
el('p2-csv').onclick=()=>{
    if(!current)return;
    const rows=[['participant','round','epsilon','epsilon_limit','delta','sigma','clip','sample_rate','steps','validation_accuracy','validation_malignant_recall','final_test_accuracy','final_test_malignant_recall','final_test_malignant_f1','mode','noise_std_sum','adaptive_reference_records','adaptive_quantile','adaptive_observed_quantile','adaptive_target_clip','adaptive_previous_clip']];
    const privacyRows=current.privacy.length?current.privacy:current.evaluations.map(e=>({round:e.round}));
    privacyRows.forEach(r=>{const e=current.evaluations.find(e=>e.round===r.round);rows.push([r.participant??'',r.round,r.epsilon??'',r.epsilon_limit??'',r.delta??'',r.sigma??'',r.clip??'',r.sample_rate??'',r.cumulative_steps??'',e?.validation.accuracy??'',e?.validation.malignant_recall??'',e?.final_test?.accuracy??'',e?.final_test?.malignant_recall??'',e?.final_test?.malignant_f1??'',current.dp.enabled?current.dp.mode:'baseline',r.noise_std_sum??(r.sigma!=null&&r.clip!=null?r.sigma*r.clip:''),r.adaptation?.reference_records??'',r.adaptation?.quantile??'',r.adaptation?.observed_quantile??'',r.adaptation?.target_clip??'',r.adaptation?.previous_clip??'']);});
    download(rows.map(r=>r.join(',')).join('\n'),'text/csv',`${scenario}_results.csv`);
};
el('p2-add').onclick=async()=>{const name=el('p2-compare-select').value;if(!name)return;const r=await fetch(`/platform/api/dashboard/${encodeURIComponent(name)}/research`);if(r.ok){compared.set(name,await r.json());renderComparison();}};
fetch('/platform/api/dashboard').then(r=>r.json()).then(rows=>{if(Array.isArray(rows))rows.forEach(row=>{const option=document.createElement('option');option.value=row.name;option.textContent=`${row.title??row.name} · ${row.status}`;el('p2-compare-select').appendChild(option);});}).catch(()=>{});
await refresh();setInterval(()=>{if(!current||current.scenario.status==='running')refresh();},5000);
