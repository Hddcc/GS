import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent / "ftd_liif" / "source"))
import train_liif_lrtfr as train_module


class StopAfterSavePath(Exception):
    pass


class ResumeSavePathTest(unittest.TestCase):
    def test_existing_checkpoint_is_preserved_without_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "face_ftd_liif"
            save_path.mkdir()
            checkpoint = save_path / "epoch-last.pth"
            checkpoint.write_bytes(b"existing checkpoint")

            with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
                with patch.object(train_module, "make_data_loaders", side_effect=StopAfterSavePath):
                    with self.assertRaises(StopAfterSavePath):
                        train_module.main({"resume": str(checkpoint)}, str(save_path))

            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")
            train_module.writer.close()

    def test_resume_does_not_apply_past_lr_milestone_twice(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_module.config = {
            "resume": "checkpoint.pth",
            "multi_step_lr": {"milestones": [2, 4], "gamma": 0.1},
        }
        train_module.log = lambda message: None
        checkpoint = {"model": {}, "optimizer": {}, "epoch": 2}

        with patch.object(train_module.torch, "load", return_value=checkpoint):
            with patch.object(train_module.models, "make", return_value=model):
                with patch.object(train_module.utils, "make_optimizer", return_value=optimizer):
                    with patch.object(model, "cuda", return_value=model):
                        _, resumed_optimizer, epoch_start, scheduler = train_module.prepare_training()

        self.assertEqual(epoch_start, 3)
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(scheduler.last_epoch, 2)
        resumed_optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 0.01)


if __name__ == "__main__":
    unittest.main()
