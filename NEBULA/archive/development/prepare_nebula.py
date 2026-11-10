"""Prepare the isolated NEBULA copy for a small local CPU demonstration.

Source inspection and syntax checks do not substitute for a Docker build.
The original upstream checkout under work/ remains unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from fix_proxy_port import corrected_config

HERE = Path(__file__).resolve().parent
REPO = HERE / 'nebula'
PIN = '525b31c674644c653117126b5a1473a5321ac240'
assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == PIN


def replace(path, before, after, count=1):
    text = path.read_text(encoding='utf-8')
    if after in text and before not in text:
        return
    if text.count(before) != count:
        raise RuntimeError(f'Unexpected upstream contents in {path}: {before!r}')
    path.write_text(text.replace(before, after), encoding='utf-8', newline='\n')


pyproject = REPO / 'pyproject.toml'
text = pyproject.read_text(encoding='utf-8')
if '[tool.uv.sources]' not in text:
    text += '''

# Project 2: CPU wheels avoid downloading the CUDA runtime.
[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
'''
    pyproject.write_text(text, encoding='utf-8', newline='\n')
assert tomllib.loads(text)['tool']['uv']['sources']['torch']['index'] == 'pytorch-cpu'

deployer = REPO / 'app/deployer.py'
replace(deployer, 'port_bindings={80: self.frontend_port, 8080: self.statistics_port},',
        'port_bindings={80: ("127.0.0.1", self.frontend_port), 8080: ("127.0.0.1", self.statistics_port)},')
replace(deployer, 'port_bindings={self.controller_port: self.controller_port},',
        'port_bindings={self.controller_port: ("127.0.0.1", self.controller_port)},')
replace(deployer, 'image="nebula-frontend",', 'image="nebula-project2-frontend",')
replace(deployer, 'image="nebula-controller",', 'image="nebula-project2-controller",')
scenarios = REPO / 'nebula/controller/scenarios.py'
replace(scenarios, 'image = "nebula-core"', 'image = "nebula-project2-core"')

# Keep one CPU thread per participant instead of multiplying four host threads.
replace(scenarios,
        'environment = {"NEBULA_LOGS_DIR": "/nebula/app/logs/", "NEBULA_CONFIG_DIR": "/nebula/app/config/"}',
        'environment = {"NEBULA_LOGS_DIR": "/nebula/app/logs/", "NEBULA_CONFIG_DIR": "/nebula/app/config/", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}')

node = REPO / 'nebula/core/node.py'
replace(node, 'elif dataset_name == "BreastCancer":\n        batch_size = 32',
        'elif dataset_name == "BreastCancer":\n        batch_size = 8')

# Small local validation sets must retain their incomplete final batch.
module = REPO / 'nebula/core/datasets/datamodule.py'
module_text = module.read_text(encoding='utf-8')
for start, end in [('    def val_dataloader(self):', '    def test_dataloader(self):'),
                   ('    def test_dataloader(self):', '    def bootstrap_dataloader(self):')]:
    begin = module_text.index(start)
    finish = module_text.index(end)
    segment = module_text[begin:finish].replace('drop_last=True', 'drop_last=False')
    module_text = module_text[:begin] + segment + module_text[finish:]
module.write_text(module_text, encoding='utf-8', newline='\n')

template = REPO / 'nebula/frontend/config/participant.json.example'
data = json.loads(template.read_text(encoding='utf-8'))
data['scenario_args'].update(federation='CFL', rounds=3, controller='127.0.0.1:5050')
data['device_args'].update(accelerator='cpu', devices=1, strategy='auto')
data['data_args'].update(dataset='BreastCancer', iid=True, num_workers=0)
data['training_args']['epochs'] = 1
data['training_args']['dp'].update(enabled=False, noise_multiplier=1.0,
                                  max_grad_norm=1.0, target_delta=1e-5, accountant='rdp')
template.write_text(json.dumps(data, indent=2), encoding='utf-8', newline='\n')

proxy_config = REPO / 'nebula/frontend/config/nebula'
proxy_config.write_text(corrected_config(proxy_config.read_text(encoding='utf-8')),
                       encoding='utf-8', newline='\n')

for relative in ['nebula/controller/start_services.sh', 'nebula/frontend/start_services.sh']:
    path = REPO / relative
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))

ignore = REPO / '.dockerignore'
ignore_text = ignore.read_text(encoding='utf-8') if ignore.exists() else ''
if '# Project 2 local environments' not in ignore_text:
    ignore_text += '''
# Project 2 local environments and generated evidence are not image build inputs.
.git
.venv
.cache
.python
.tools
__pycache__
**/__pycache__
*.pyc
app/logs
app/databases
app/config
app/certs
project2-evidence
'''
    ignore.write_text(ignore_text, encoding='utf-8', newline='\n')

for path in [deployer, scenarios, node, module]:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
assert 'batch_size = 8' in node.read_text(encoding='utf-8')
assert 'drop_last=True' not in module.read_text(encoding='utf-8').split('    def val_dataloader(self):')[1].split('    def bootstrap_dataloader(self):')[0]
revision = hashlib.sha256()
for path in [pyproject, deployer, scenarios, node, module, template, proxy_config, ignore,
             REPO / 'nebula/controller/start_services.sh',
             REPO / 'nebula/frontend/start_services.sh', HERE / 'Dockerfile.cpu']:
    revision.update(path.name.encode('utf-8'))
    revision.update(path.read_bytes())
(REPO / '.project2-revision').write_text(revision.hexdigest() + '\n', encoding='ascii')
print('NEBULA CPU/local-demo preparation and Python syntax checks passed.')

# Project 2 privacy/results integration
subprocess.check_call([__import__("sys").executable, "-B", str(HERE / "enhance_nebula.py")])
subprocess.check_call([__import__("sys").executable, "-B", str(HERE / "build_update_manifest.py")])
