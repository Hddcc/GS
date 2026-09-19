"""Package the read-only analyzer and launcher into one ASCII-only upload."""

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('Commit implementation before packaging.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).strip()
    name = 'face_formal_paired_audit_bundle_20260919'
    output = args.output_dir / (name + '.zip')
    if output.exists():
        raise FileExistsError(output)
    files = {}
    for path in ('audit_face_formal_paired.py', 'run_face_formal_paired_audit_20260919.sh'):
        files[path] = (root / path).read_bytes().replace(b'\r\n', b'\n')
    files['IMPLEMENTATION_COMMIT'] = commit + b'\n'
    files['FILES_SHA256'] = ''.join(
        '{}  {}\n'.format(hashlib.sha256(data).hexdigest(), filename)
        for filename, data in sorted(files.items())).encode('ascii')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename, data in sorted(files.items()):
            assert filename.isascii()
            archive.writestr(name + '/' + filename, data)
    print('ZIP={}'.format(output))
    print('SHA256={}'.format(hashlib.sha256(output.read_bytes()).hexdigest()))
    print('IMPLEMENTATION_COMMIT={}'.format(commit.decode('ascii')))


if __name__ == '__main__':
    main()
