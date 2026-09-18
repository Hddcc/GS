import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path


FILES = (
    'models/models.py', 'models/edsr.py', 'models/gaussian.py', 'models/mlp.py',
    'models/gaussian_face_frequency_residual.py',
    'models/gaussian_face_scale_gated_frequency.py',
    'datasets/__init__.py', 'datasets/datasets.py', 'datasets/image_folder.py',
    'datasets/wrappers.py', 'utils.py', 'test.py', 'train_gaussian.py',
    'scale_gated_frequency_training.py', 'test_face_scale_gated_frequency.py',
    'prepare_face_scale_gated_frequency.py', 'evaluate_face_scale_gated_frequency.py',
    'run_face_scale_gated_frequency.sh',
    'configs/train/face/train_face_gaussian_baseline_seed1.yaml',
)
MODEL_INIT = '''from .models import register, make
from . import edsr, gaussian, mlp
from . import gaussian_face_frequency_residual
from . import gaussian_face_scale_gated_frequency
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
    if status.strip():
        raise RuntimeError('Commit and push the implementation before packaging.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    bundle = 'face_scale_gated_frequency_bundle_20260918'
    output = args.output_dir / (bundle + '.zip')
    if output.exists():
        raise FileExistsError('Refusing to replace existing bundle: ' + str(output))
    payload = {}
    for name in FILES:
        payload['runtime/' + name] = (root / name).read_bytes().replace(b'\r\n', b'\n')
    payload['runtime/models/__init__.py'] = MODEL_INIT.encode('ascii')
    payload['launch_face_scale_gated_frequency.sh'] = (
        root / 'launch_face_scale_gated_frequency.sh'
    ).read_bytes().replace(b'\r\n', b'\n')
    payload['IMPLEMENTATION_COMMIT'] = (commit + '\n').encode('ascii')
    for name in ('LICENSE', 'LICENSE.txt'):
        if (root / name).is_file():
            payload[name] = (root / name).read_bytes()
    manifest = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n'
                       for name, data in sorted(payload.items()))
    payload['FILES_SHA256'] = manifest.encode('ascii')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            assert name.isascii() and not name.endswith('.md')
            archive.writestr(bundle + '/' + name, data)
    print('ZIP=' + str(output))
    print('SHA256=' + hashlib.sha256(output.read_bytes()).hexdigest())
    print('IMPLEMENTATION_COMMIT=' + commit)


if __name__ == '__main__':
    main()
