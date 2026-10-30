import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from prepare_configs import prepare_fplia


ROOT = Path(__file__).resolve().parent


class PrepareFPLIAConfigTest(unittest.TestCase):
    def test_lia_model_arguments_match_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            addon = Path(directory)
            configs = addon / "fplia/source/configs"
            configs.mkdir(parents=True)
            shutil.copyfile(ROOT / "fplia/source/configs/lia-swinir.yaml",
                            configs / "lia-swinir.yaml")
            args = SimpleNamespace(
                addon=addon, stage="lia-swinir", train=addon / "train",
                val=addon / "val", batch_size=1, epochs=1000,
                lia_checkpoint=None, vae_checkpoint=None,
                diffusion_checkpoint=None,
            )
            prepare_fplia(args)
            generated = yaml.safe_load(
                (addon / "fplia/config/face_lia-swinir.yaml").read_text(encoding="utf-8")
            )

        source = ROOT / "fplia/source/diffusion/models/lia.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        model_class = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "LIAModel")
        constructor = next(node for node in model_class.body
                           if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        accepted = {arg.arg for arg in constructor.args.args[1:]}
        model_args = generated["model"]["params"]
        self.assertFalse(set(model_args) - accepted)
        self.assertEqual(generated["data"]["params"]["train"]["wrapper"]["params"]["sample_q"], 1024)


if __name__ == "__main__":
    unittest.main()
