"""Preserve the browser's port in NEBULA links without rebuilding its images."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def corrected_config(text):
    old = 'proxy_set_header Host $host;'
    new = 'proxy_set_header Host $http_host;'
    if text.count(old) == 1:
        text = text.replace(old, new)
    elif text.count(new) != 1:
        raise ValueError('Unexpected Host configuration. No automatic change is safe.')
    # These headers already appear explicitly below the include in this template.
    text = text.replace('        include proxy_params;\n', '')
    text = text.replace('proxy_set_header X-Forwarded-Host $host;',
                        'proxy_set_header X-Forwarded-Host $http_host;')
    # $server_port is the internal container port 80, not the published port.
    text = text.replace('        proxy_set_header X-Forwarded-Port $server_port;\n', '')
    if 'include proxy_params;' in text:
        raise ValueError('Unexpected proxy include indentation. No file was changed.')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--reload', action='store_true', help='Validate and reload the running frontend inside Ubuntu.')
    parser.add_argument('--container', help='Override the default <USER>_nebula-frontend container.')
    args = parser.parse_args()
    if args.reload and not sys.platform.startswith('linux'):
        raise SystemExit('--reload must run in Ubuntu/WSL.')
    project = args.project.expanduser().resolve()
    config = project / 'nebula/frontend/config/nebula'
    if not config.is_file():
        raise SystemExit(f'NEBULA configuration not found: {config}')
    before = config.read_text(encoding='utf-8')
    after = corrected_config(before)
    backup = config.with_name('nebula.before-port-fix')
    changed = before != after
    if changed:
        if not backup.exists():
            shutil.copyfile(config, backup)
        config.write_text(after, encoding='utf-8', newline='\n')
    try:
        if args.reload:
            user = os.environ.get('USER')
            if not args.container and not user:
                raise RuntimeError('USER is unavailable. Supply --container explicitly.')
            container = args.container or f'{user}_nebula-frontend'
            subprocess.run(['docker', 'exec', container, 'nginx', '-t'], check=True)
            subprocess.run(['docker', 'exec', container, 'nginx', '-s', 'reload'], check=True)
    except Exception:
        if changed:
            config.write_text(before, encoding='utf-8', newline='\n')
        raise
    print('Proxy port configuration corrected.' if changed else 'Proxy port configuration already correct.')
    if args.reload:
        print('Nginx validation and reload completed. Refresh http://127.0.0.1:6060/platform')
    else:
        print('Running frontend was not reloaded. Apply --reload inside Ubuntu for an active installation.')


if __name__ == '__main__':
    main()
