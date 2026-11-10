"""Plot the recorded feasibility measurements without rerunning or selecting runs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
paths = [root / 'results/20261026_baseline', root / 'results/20261026_dp_sigma1']
records = [json.loads((path / 'summary.json').read_text(encoding='utf-8')) for path in paths]
manifests = [json.loads((path / 'data_manifest.json').read_text(encoding='utf-8')) for path in paths]
assert manifests[0] == manifests[1]
for record, path in zip(records, paths):
    # Annotate the measured interval, without changing any measured number.
    record['wall_clock_scope'] = 'federated loop including public validation, privacy queries and console output; excludes import/setup and final test'
    record['local_optimizer_state'] = 'Adam moments persist between rounds in this standalone pilot'
    if record['mode'] == 'baseline':
        record['privacy_scope'] = 'no DP bound, non-private baseline'
    (path / 'summary.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), layout='constrained')
colors = ['#2a6f9e', '#d28a36']
labels = ['Baseline', 'DP-SGD']
bars = axes[0].bar(labels, [r['test_metrics']['accuracy'] * 100 for r in records], color=colors, width=.55)
axes[0].bar_label(bars, fmt='%.2f%%', padding=3)
axes[0].set(ylim=(0, 108), ylabel='Test accuracy (%)', title='Public test split: 114 records')
bars = axes[1].bar(labels, [r['training_wall_seconds'] for r in records], color=colors, width=.55)
axes[1].bar_label(bars, fmt='%.3f s', padding=3)
axes[1].set(ylim=(0, max(r['training_wall_seconds'] for r in records)*1.22), ylabel='Federated-loop wall time (s)', title='Three rounds, three simulated clients')
fig.suptitle('Standalone CPU feasibility pilot, not a NEBULA container experiment', fontsize=12)
fig.supxlabel('One run per setting. DP-SGD: clipping C=1, noise multiplier=1, nominal epsilon=3.80 at delta=0.00001.\nTiming includes public validation and privacy accounting. No clinical validation.', fontsize=8)
fig.savefig(root / 'results/pilot_comparison.png', dpi=180)
print('Created results/pilot_comparison.png from the two recorded runs.')
