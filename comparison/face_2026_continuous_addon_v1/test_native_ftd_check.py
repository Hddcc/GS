import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from evaluate_ftd import coord_axis, image_tensor
from native_ftd_check import compare_rows, make_dataset


class NativeFTDCheckTest(unittest.TestCase):
    def test_native_wrapper_matches_evaluation_inputs(self):
        images = ROOT.parents[3] / "comparison_methods_20261002/datasets/face/CelebA/test/HR"
        source = ROOT / "ftd_liif/source"
        for scale in (3, 3.25, 3.5, 3.75, 4):
            with self.subTest(scale=scale):
                dataset, names = make_dataset(source, images, scale, 1)
                native = dataset[0]
                with Image.open(images / names[0]) as image:
                    width, height = image.size
                    low_h = math.floor(height / scale + 1e-9)
                    low_w = math.floor(width / scale + 1e-9)
                    out_h = round(low_h * scale)
                    out_w = round(low_w * scale)
                    image = image.crop((0, 0, out_w, out_h))
                    inp = image_tensor(image.resize((low_w, low_h), Image.Resampling.BICUBIC))
                    gt = image_tensor(image)
                self.assertTrue(torch.equal(native["inp"], inp))
                self.assertTrue(torch.equal(native["gt"], gt))
                self.assertTrue(torch.allclose(native["coordx"], coord_axis(out_h, "cpu")))
                self.assertTrue(torch.allclose(native["coordy"], coord_axis(out_w, "cpu")))

    def test_per_image_comparison_detects_a_mismatch(self):
        row = {"image": "a.png", "scale": 3.5, "crop_border": 4,
               "psnr_y": 27.0, "ssim_y": 0.8}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            compare_rows([row], path)
            altered = dict(row, psnr_y=28.0)
            with self.assertRaisesRegex(ValueError, "does not match"):
                compare_rows([altered], path)


if __name__ == "__main__":
    unittest.main()
