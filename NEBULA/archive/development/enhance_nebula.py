"""Apply the Project 2 privacy/results integration to the prepared checkout."""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parent / "nebula"


def change(relative, before, after):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if after in text:
        return
    if text.count(before) != 1:
        raise ValueError(f"Unexpected source fragment in {relative}: {before[:80]}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8", newline="\n")


change("nebula/core/training/lightning_dp.py",
       "            clipping=dp_config[\"clipping\"],\n",
       "            clipping=dp_config[\"clipping\"],\n            mode=dp_config.get('mode', 'fixed'),\n            noise_end=dp_config.get('noise_end', 0.7),\n            target_epsilon=dp_config.get('target_epsilon', 8.0),\n            total_rounds=int(self.config.participant['scenario_args']['rounds']),\n            config=self.config,\n")
change("nebula/core/training/lightning_dp.py", "            state.extras[\"dataloader\"] = train_dataloader\n",
       "            state.extras['epochs'] = self.epochs\n            state.extras['round_index'] = self.round\n            state.extras[\"dataloader\"] = train_dataloader\n")
change("nebula/core/training/lightning_dp.py", "            config=self.config,\n",
       "            config=self.config,\n            **{key: dp_config[key] for key in (\"adaptive_quantile\", \"adaptive_smoothing\",\n               \"adaptive_min_clip\", \"adaptive_max_clip\", \"adaptive_max_records\") if key in dp_config},\n")
change("nebula/core/training/lightning_dp.py", '            state.extras["dataloader"] = train_dataloader\n',
       '''            state.extras["dataloader"] = train_dataloader
            if self._dp_plugin.settings["mode"] == "adaptive":
                public = getattr(self.datamodule, "public_validation", None)
                if public is None:
                    raise ValueError("Adaptive mode requires the designated public calibration dataset")
                state.extras["public_reference_dataset"] = public
                state.extras["public_reference_is_public"] = True
''')
change("nebula/core/training/lightning_dp.py", "            return loss, accuracy, train_accuracy",
       "            # Only public validation is released. Private loss/train accuracy stay local.\n            return None, accuracy, None")
change("nebula/core/datasets/breast_cancer/breast_cancer.py", "        test_size: float = 0.2,\n",
       "        test_size: float = 0.2,\n        research_protocol: bool = False,\n")
change("nebula/core/datasets/breast_cancer/breast_cancer.py", "        self.test_size = float(test_size)",
       "        self.test_size = float(test_size)\n        self.research_protocol = research_protocol")
change("nebula/core/datasets/breast_cancer/breast_cancer.py", "    def initialize_dataset(self):\n        if self.train_set",
       "    def initialize_dataset(self):\n        if self.research_protocol:\n            from nebula.core.datasets.breast_cancer.research_data import write_research_partitions\n            return write_research_partitions(self)\n        if self.train_set")
change("nebula/controller/scenarios.py", "            dataset = BreastCancerDataset(\n                num_classes=2,",
       "            dataset = BreastCancerDataset(\n                research_protocol=self.scenario.federation == 'CFL',\n                num_classes=2,")
relative = "nebula/controller/scenarios.py"
path = ROOT / relative
text = path.read_text(encoding="utf-8")
start = text.index("            # Project 2 keeps every privacy setting") if "# Project 2 keeps every privacy setting" in text else text.index("            if isinstance(self.scenario.dp, dict):")
end = text.index("            feature_squeezing = (", start)
replacement = '''            # Project 2 keeps every privacy setting in the exported participant config.
            from nebula.core.training.privacy_policy import validate_dp_config
            participant_config["training_args"]["dp"] = validate_dp_config(self.scenario.dp)
            if participant_config["training_args"]["dp"]["enabled"]:
                if self.scenario.dataset != "BreastCancer" or self.scenario.federation != "CFL":
                    raise ValueError("Project 2 DP requires BreastCancer with CFL")
                if (self.scenario.attack_params or {}).get("attacks", "No Attack") != "No Attack" or (self.scenario.reputation or {}).get("enabled", False):
                    raise ValueError("Disable attacks and reputation for the controlled DP experiment")
                if self.scenario.with_sa or self.scenario.with_trustworthiness or self.scenario.mobility:
                    raise ValueError("Disable situational awareness, trustworthiness and mobility for the controlled DP experiment")
                if (self.scenario.adversarial_training or {}).get("enabled", False) or (self.scenario.feature_squeezing or {}).get("enabled", False):
                    raise ValueError("Disable additional training defenses for the controlled DP experiment")
                if participant_config["training_args"]["dp"]["secure_mode"]:
                    import importlib.util
                    if importlib.util.find_spec("torchcsprng") is None:
                        raise ValueError("CSPRNG mode requires torchcsprng in the CPU image. Experimental mode is explicitly labeled in the report.")
'''
if text[start:end] != replacement:
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8", newline="\n")
change("nebula/core/node.py", "    dataset.log_partition()",
       "    import json\n    from pathlib import Path\n    research_root = Path(config.participant['tracking_args']['config_dir'])\n    research = (research_root / 'research_manifest.json').exists()\n    model.research_public_evaluation = research\n    if research:\n        # Evaluation membership must not depend on private training labels.\n        dataset.local_test_indices = list(range(len(dataset.test_set)))\n    if not (research and config.participant['training_args']['dp']['enabled']):\n        dataset.log_partition()")
change("nebula/core/node.py", "    trainer = None\n    trainer_str =",
       "    if research:\n        datamodule.public_validation = handler(str(research_root / 'public_validation.h5'), 'test', config=config)\n\n    trainer = None\n    trainer_str =")
change("nebula/core/datasets/datamodule.py", "    def setup(self, stage=None):\n        if stage",
       "    def setup(self, stage=None):\n        if stage in (None, 'fit') and getattr(self, 'public_validation', None) is not None:\n            self.data_train = ChangeableSubset(self.train_set, self.train_set_indices)\n            self.data_val = self.public_validation\n            self.model_weight = len(self.data_train)\n            if stage == 'fit':\n                return\n            stage = 'test'\n        if stage")
change("nebula/core/datasets/datamodule.py", "        return DataLoader(\n            self.data_train,\n            batch_size=self.batch_size,\n            shuffle=True,\n            num_workers=self.num_workers,\n            drop_last=True,\n            pin_memory=False,\n        )",
       "        loader = DataLoader(\n            self.data_train, batch_size=self.batch_size, shuffle=True,\n            num_workers=self.num_workers,\n            drop_last=not hasattr(self, 'public_validation'), pin_memory=False,\n        )\n        if hasattr(self, 'public_validation'):\n            # Match Poisson sampling in baseline, fixed DP and dynamic DP.\n            from opacus.data_loader import DPDataLoader\n            return DPDataLoader.from_data_loader(loader)\n        return loader")
change("nebula/core/models/nebulamodel.py", "        y_pred_classes = torch.argmax(y_pred, dim=1).detach()",
       "        if phase == 'Train' and getattr(self, 'research_public_evaluation', False):\n            return\n        y_pred_classes = torch.argmax(y_pred, dim=1).detach()")
change("nebula/core/models/nebulamodel.py", "        if phase == \"Train\":\n            output = self.train_metrics.compute()",
       "        if phase == 'Train' and getattr(self, 'research_public_evaluation', False):\n            self._latest_train_metrics = {}\n            return\n        if phase == \"Train\":\n            output = self.train_metrics.compute()")
change("nebula/core/models/nebulamodel.py", "            loss = self.criterion(y_pred, y)\n\n        self.process_metrics",
       "            loss = self.criterion(y_pred, y) if len(y) else y_pred.sum()\n\n        self.process_metrics")
change("nebula/core/training/lightning.py", "        except Exception as e:\n            logging_training.error(f\"Error training model: {e}\")\n            logging_training.error(traceback.format_exc())",
       "        except Exception as e:\n            logging_training.error(f\"Error training model: {e}\")\n            logging_training.error(traceback.format_exc())\n            if self.config.participant['training_args'].get('dp', {}).get('enabled'):\n                from nebula.core.training.research_results import append_record\n                append_record(self.config, 'privacy_error', {'round': self.round + 1, 'error': str(e)})\n                raise")
change("nebula/core/engine.py", "                await self.get_round_lock().acquire_async()\n\n                print_msg_box(",
       "                from nebula.core.training.research_results import record_server_round\n                await asyncio.to_thread(record_server_round, self.trainer, self.round,\n                                        time.perf_counter() - project2_round_started,\n                                        self.round + 1 == self.total_rounds)\n                await self.get_round_lock().acquire_async()\n\n                print_msg_box(")
# Record monotonic round timing independently of event attribute names.
change("nebula/core/engine.py", "                current_time = time.time()\n                print_msg_box(",
       "                current_time = time.time()\n                project2_round_started = time.perf_counter()\n                print_msg_box(")
change("nebula/frontend/static/js/deployment/ui-controls.js", '                case "percentage":', '                case "percent":')
change("nebula/controller/controller.py", '''    except subprocess.CalledProcessError as e:
        logging.exception(f"Error docker-compose up: {e}")
        return''', '''    except Exception as e:
        logging.exception(f"Scenario startup failed: {e}")
        from pathlib import Path
        from datetime import datetime
        failure = {"kind": "startup_error", "error": str(e)}
        error_path = Path(scenarioManagement.log_dir) / scenarioManagement.scenario_name / "scenario_startup_error.jsonl"
        with error_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure) + "\\n")
        await update_scenario(scenario_name=scenarioManagement.scenario_name,
            start_time=scenarioManagement.start_date_scenario,
            end_time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            scenario=scenario_data, status="failed", role=role, username=user)
        raise HTTPException(status_code=500, detail=f"Scenario startup failed: {e}") from e''')
change("nebula/frontend/templates/deployment.html", '                        <h5 class="step-title">Noise multiplier</h5>', '''                        <p>Project 2: protección de registros mediante recorte, ruido y contabilidad acumulada.</p>
                        <label for="dpMode">Modo</label>
                        <select id="dpMode" class="form-control"><option value="fixed">DP fija</option><option value="adaptive">DP adaptativa con referencia pública</option></select>
                        <label for="dpEpsilon">Presupuesto máximo ε por cliente</label>
                        <input id="dpEpsilon" class="form-control" type="number" min="0.25" max="100" step="0.25" value="8">
                        <small>ε menor implica una cota más estricta para el mismo δ. No expresa un porcentaje de privacidad.</small>
                        <br><label for="dpDelta">δ, término aditivo de la garantía</label>
                        <input id="dpDelta" class="form-control" type="number" min="0.000000001" max="0.01" step="0.000001" value="0.00001">
                        <p>El multiplicador σ se calibra antes de entrenar y permanece constante. En modo adaptativo, C responde a los gradientes del modelo actual en los 91 registros públicos. El ruido aplicado cambia junto con C.</p>
                        <div id="dp-adaptive-settings" style="display:none">
                            <label for="dpAdaptiveQuantile">Cuantil de referencia, 0.5 = mediana</label>
                            <input id="dpAdaptiveQuantile" class="form-control" type="number" min="0.01" max="0.99" step="0.05" value="0.5">
                            <label for="dpAdaptiveSmoothing">Peso del nuevo valor, 1 = ajuste inmediato</label>
                            <input id="dpAdaptiveSmoothing" class="form-control" type="number" min="0.01" max="1" step="0.05" value="0.5">
                            <label for="dpAdaptiveMinClip">C mínimo</label>
                            <input id="dpAdaptiveMinClip" class="form-control" type="number" min="0.001" step="0.1" value="0.1">
                            <label for="dpAdaptiveMaxClip">C máximo</label>
                            <input id="dpAdaptiveMaxClip" class="form-control" type="number" min="0.001" step="0.5" value="5">
                            <small>El ajuste consulta la referencia pública y el modelo recibido, sin leer gradientes privados ni el test final.</small>
                        </div>
                        <h5 class="step-title">Límite C, fijo o inicial para adaptación</h5>''')
# Remove the obsolete relative-profile input from a freshly prepared checkout.
path = ROOT / "nebula/frontend/templates/deployment.html"
text = path.read_text(encoding="utf-8")
start = text.find('                        <div class="form-check form-check-inline">\n                            <input type="number" class="form-control" id="dpNoiseMultiplier"')
if start >= 0:
    end = text.index('                        <h5 class="step-title">Max Grad Norm</h5>', start)
    text = text[:start] + text[end:]
    text = text.replace('<h5 class="step-title">Max Grad Norm</h5>', '<h5 class="step-title">Límite C, fijo o inicial para adaptación</h5>')
    # The adaptive heading is already inserted above the removed input.
    heading = '                        <h5 class="step-title">Límite C, fijo o inicial para adaptación</h5>\n'
    text = text.replace(heading + heading, heading)
    path.write_text(text, encoding="utf-8", newline="\n")
change("nebula/frontend/templates/deployment.html", '''                        <small class="form-text text-muted">
                            Other DP parameters are still taken from
                            <code>nebula/frontend/config/participant.json.example</code>.
                        </small>''', '''                        <label><input type="checkbox" id="dpSecureMode"> CSPRNG para ruido y muestreo, requiere torchcsprng instalado</label>
                        <p>Sin CSPRNG, el panel identificará el experimento como modo experimental. La contabilidad condiciona una partición fija y pública. El protocolo requiere BreastCancer, CFL y tres trainers.</p>''')
change("nebula/frontend/templates/monitor.html", '{{ super() }}', '''{{ super() }}
{% if scenario %}<div class="container" style="padding-top:100px"><a class="btn btn-info" href="/platform/dashboard/{{ scenario_name }}/research">Privacidad y resultados · Project 2</a></div>{% endif %}''')
change("nebula/frontend/app.py", '@app.post("/platform/dashboard/{scenario_name}/node/update")', '''@app.get("/platform/dashboard/{scenario_name}/research", response_class=HTMLResponse)
async def project2_research_page(scenario_name: str, request: Request, session: dict = Depends(get_session)):
    if "user" not in session:
        raise HTTPException(status_code=401)
    scenario = await get_scenario_by_name(scenario_name)
    if not scenario:
        raise HTTPException(status_code=404)
    if session.get("role") != "admin" and str(scenario.get("username", "")).upper() != str(session["user"]).upper():
        raise HTTPException(status_code=403)
    return templates.TemplateResponse("research.html", {"request": request, "scenario_name": scenario_name,
        "scenario": scenario, "user_logged_in": session.get("user")})


@app.get("/platform/api/dashboard/{scenario_name}/research", response_class=JSONResponse)
async def project2_research_data(scenario_name: str, session: dict = Depends(get_session)):
    if "user" not in session:
        raise HTTPException(status_code=401)
    scenario = await get_scenario_by_name(scenario_name)
    if not scenario:
        raise HTTPException(status_code=404)
    if session.get("role") != "admin" and str(scenario.get("username", "")).upper() != str(session["user"]).upper():
        raise HTTPException(status_code=403)
    from nebula.frontend.research_view import research_payload
    logs = FileUtils.check_path(settings.log_dir, scenario_name)
    configs = FileUtils.check_path(settings.config_dir, scenario_name)
    return JSONResponse(research_payload(logs, configs, scenario))


@app.post("/platform/dashboard/{scenario_name}/node/update")''')
template = ROOT / "nebula/frontend/config/participant.json.example"
data = json.loads(template.read_text(encoding="utf-8"))
data["training_args"]["dp"].update(mode="fixed", noise_end=0.7, target_epsilon=8.0,
                                   target_delta=1e-5, accountant="rdp", adaptive_quantile=0.5,
                                   adaptive_smoothing=0.5, adaptive_min_clip=0.1,
                                   adaptive_max_clip=5.0, adaptive_max_records=91)
template.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
for relative in ("nebula/core/node.py", "nebula/core/engine.py", "nebula/core/datasets/datamodule.py",
                 "nebula/core/models/nebulamodel.py", "nebula/core/training/lightning_dp.py",
                 "nebula/core/training/lightning.py", "nebula/controller/scenarios.py"):
    ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
print("Project 2 integration source patches applied")
