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
from diagnose_ftd_inference import diagnose, evaluation_coords, native_coords
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

    def test_inference_diagnostic_separates_coordinate_paths(self):
        class CoordinateModel(torch.nn.Module):
            def forward(self, inp, coord, coordx, coordy, cellx, celly):
                return coord[..., :1].permute(0, 3, 1, 2).expand(-1, 3, -1, -1)

        axis = coord_axis(16, "cpu")
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
        sample = {
            "coord": torch.stack((grid_x, grid_y), dim=-1),
            "coordx": axis, "coordy": axis,
            "cellx": torch.full_like(axis, 2 / 16),
            "celly": torch.full_like(axis, 2 / 16),
        }
        native = native_coords(sample, "cpu")
        evaluated = list(evaluation_coords(16, 16, "cpu"))
        evaluated[0] = evaluated[0] + 0.01
        scores = diagnose(CoordinateModel(), torch.zeros(1, 3, 8, 8),
                          torch.zeros(1, 3, 16, 16), native, evaluated,
                          2.0, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1))
        self.assertEqual(scores["native_repeat_max_pixel_delta"], 0)
        self.assertGreater(scores["native_evaluated_max_pixel_delta"], 0)
        self.assertGreater(scores["coordinate_max_delta"], 0)


if __name__ == "__main__":
    unittest.main()
