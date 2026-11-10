"""Install a checksum-verified uv wheel's Linux executable in the project only."""
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import urllib.request
import zipfile

VERSION = '0.8.22'


def main():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise SystemExit('This installer supports Linux/WSL on x86_64.')
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'https://pypi.org/pypi/uv/{VERSION}/json', timeout=60) as stream:
        metadata = json.load(stream)
    candidates = [item for item in metadata['urls']
                  if not item.get('yanked') and item['filename'].endswith('.whl')
                  and 'manylinux' in item['filename'] and 'x86_64' in item['filename']]
    if len(candidates) != 1:
        raise SystemExit('Unexpected uv wheel list. No executable was installed.')
    wheel = candidates[0]
    if not wheel['url'].startswith('https://files.pythonhosted.org/'):
        raise SystemExit('Unexpected package host.')
    with urllib.request.urlopen(wheel['url'], timeout=120) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != wheel['digests']['sha256']:
        raise SystemExit('Wheel checksum mismatch.')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [name for name in archive.namelist() if name.endswith('/uv') and not name.endswith('/')]
        if len(names) != 1:
            raise SystemExit('Unexpected executable layout.')
        executable = destination / 'uv'
        executable.write_bytes(archive.read(names[0]))
        executable.chmod(0o755)
    # Retain the original wheel, including its license and package metadata.
    (destination / wheel['filename']).write_bytes(data)
    (destination / 'uv-provenance.json').write_text(json.dumps({
        'version': VERSION, 'wheel': wheel['filename'], 'source': wheel['url'],
        'sha256': wheel['digests']['sha256'],
    }, indent=2), encoding='utf-8')
    print(f'Project-local uv {VERSION} installed with a verified wheel checksum.')


if __name__ == '__main__':
    main()
