import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PANELS = (
    ('bicubic.png', 'Bicubic'),
    ('baseline.png', 'GaussianSR'),
    ('frequency.png', 'Frequency only'),
    ('edge.png', 'Edge only'),
    ('full.png', 'Ours'),
    ('ground_truth.png', 'Ground Truth'),
)


def text_size(draw, text, font):
    if hasattr(draw, 'textbbox'):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def make_comparison(sample_dir, output_name, label_height, gap):
    paths = [(sample_dir / filename, label) for filename, label in PANELS]
    missing = [str(path) for path, _ in paths if not path.is_file()]
    if missing:
        return False, 'missing {}'.format(', '.join(missing))

    images = [Image.open(path).convert('RGB') for path, _ in paths]
    target_size = images[-1].size
    if any(image.size != target_size for image in images):
        images = [image.resize(target_size, Image.Resampling.BICUBIC) for image in images]

    width, height = target_size
    canvas_width = len(images) * width + (len(images) - 1) * gap
    canvas = Image.new('RGB', (canvas_width, height + label_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for column, (image, (_, label)) in enumerate(zip(images, paths)):
        left = column * (width + gap)
        canvas.paste(image, (left, label_height))
        label_width, label_text_height = text_size(draw, label, font)
        text_x = left + max(0, (width - label_width) // 2)
        text_y = max(0, (label_height - label_text_height) // 2)
        draw.text((text_x, text_y), label, fill='black', font=font)

    output_path = sample_dir / output_name
    canvas.save(output_path)
    return True, str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Compose labeled face SR comparison boards from exported PNG files.'
    )
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-name', default='comparison.png')
    parser.add_argument('--label-height', type=int, default=24)
    parser.add_argument('--gap', type=int, default=4)
    args = parser.parse_args()

    root = Path(args.input_root)
    if not root.is_dir():
        raise FileNotFoundError('Input directory not found: {}'.format(root))
    if args.label_height < 1 or args.gap < 0:
        raise ValueError('--label-height must be positive and --gap must be non-negative.')

    sample_dirs = sorted({path.parent for path in root.rglob('ground_truth.png')})
    if not sample_dirs:
        raise RuntimeError('No exported sample directories found under {}'.format(root))

    completed = 0
    for sample_dir in sample_dirs:
        ok, message = make_comparison(
            sample_dir, args.output_name, args.label_height, args.gap
        )
        if ok:
            completed += 1
            print(message)
        else:
            print('SKIP {}: {}'.format(sample_dir, message))
    print('COMPLETED: {}/{} comparison boards'.format(completed, len(sample_dirs)))


if __name__ == '__main__':
    main()
