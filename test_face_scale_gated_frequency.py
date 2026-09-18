import argparse
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

import models
from scale_gated_frequency_training import (
    initialize_from_baseline, optimizer_groups, set_training_stage,
)
from utils import make_coord


def check_trainer_integration(path):
    import train_gaussian as trainer
    trainer.training_seed = 1
    trainer.log = lambda message: None
    adaptation = {'checkpoint': str(path), 'backbone_lr': 1e-6,
                  'residual_lr': 1e-4, 'freeze_epochs': 3}
    trainer.config = {'model': model_spec(True), 'baseline_adaptation': adaptation,
                      'optimizer': {'name': 'adam', 'args': {'lr': 1e-6}}}
    with patch.object(torch.nn.Module, 'cuda', lambda self: self):
        model, optimizer, epoch, scheduler, random_state, best = trainer.prepare_training()
        assert epoch == 1 and len(optimizer.param_groups) == 2
        assert trainer.config['baseline_source']['source_epoch'] == 1350
        resume_path = Path(path).parent / 'resume.pth'
        checkpoint = {
            'model': dict(model_spec(True), sd=model.state_dict()),
            'optimizer': {'name': 'adam', 'args': {'lr': 1e-6}, 'sd': optimizer.state_dict()},
            'epoch': 3, 'seed': 1, 'baseline_source': trainer.config['baseline_source'],
            'baseline_adaptation': adaptation,
        }
        torch.save(checkpoint, resume_path)
        trainer.config['resume'] = str(resume_path)
        restored, restored_opt, epoch, *_ = trainer.prepare_training()
        assert epoch == 4 and len(restored_opt.param_groups) == 2
        trainer.config['baseline_adaptation'] = dict(adaptation, freeze_epochs=2)
        try:
            trainer.prepare_training()
        except ValueError:
            pass
        else:
            raise AssertionError('Changed resume protocol was accepted.')


def model_spec(gated=None):
    args = {
        'encoder_spec': {'name': 'edsr-baseline', 'args': {
            'n_resblocks': 2, 'n_feats': 16, 'no_upsampling': True,
            'use_pretrained': False}},
        'dec_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [16]}},
        'kernel_size': 5, 'hidden_dim': 16,
    }
    name = 'gaussian-splatter'
    if gated is not None:
        name = 'gaussian-splatter-face-scale-gated-frequency'
        args.update(use_scale_gate=gated, residual_feature_dim=8,
                    residual_hidden_dim=16, gate_hidden_dim=8)
    return {'name': name, 'args': args}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1)
    baseline = models.make(model_spec()).to(args.device).eval()
    checkpoint = {'model': dict(model_spec(), sd=baseline.state_dict()),
                  'seed': 1, 'epoch': 1350}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'baseline.pth'
        torch.save(checkpoint, path)
        if args.device == 'cpu':
            check_trainer_integration(path)
        for gated in (False, True):
            candidate = models.make(model_spec(gated)).to(args.device).eval()
            metadata = initialize_from_baseline(candidate, path, 1)
            assert metadata['source_epoch'] == 1350
            inp = torch.randn(1, 3, 7, 9, device=args.device)
            coord = make_coord((14, 18)).to(args.device)[::9].unsqueeze(0)
            cell = torch.ones_like(coord) * coord.new_tensor([2 / 14, 2 / 18])
            for value in (2., 2.5, 3., 3.5, 4., 8.):
                scale = coord.new_tensor([value])
                with torch.no_grad():
                    expected = baseline(inp, coord, scale, cell)
                    actual = candidate(inp, coord, scale, cell)
                error = (actual - expected).abs().max().item()
                assert error <= 1e-6, error
                print('gate={}, scale={}: initial error={:.3e}'.format(gated, value, error))
            baseline.train()
            candidate.train()
            with torch.no_grad():
                torch.manual_seed(123)
                expected = baseline(inp, coord, coord.new_tensor([2.5]), cell)
                torch.manual_seed(123)
                actual = candidate(inp, coord, coord.new_tensor([2.5]), cell)
            assert torch.allclose(expected, actual, atol=1e-6, rtol=0)
            baseline.eval()
            groups = optimizer_groups(candidate, 1e-6, 1e-4)
            optimizer = torch.optim.Adam(groups)
            original = {name: p.detach().clone() for name, p in candidate.named_parameters()
                        if name in checkpoint['model']['sd']}
            assert set_training_stage(candidate, 1, 3) == 'residual-only'
            for _ in range(3):
                optimizer.zero_grad()
                prediction = candidate(inp, coord, coord.new_tensor([2.5]), cell)
                prediction.square().mean().backward()
                optimizer.step()
            assert all(torch.equal(original[name], p) for name, p in
                       candidate.named_parameters() if name in original)
            for module in (candidate.frequency_encoder, candidate.residual_head,
                           candidate.scale_gate):
                if module is not None:
                    assert any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in module.parameters())
            assert candidate.last_rgb_residual.abs().mean() > 0
            state = copy.deepcopy(optimizer.state_dict())
            restored = models.make(model_spec(gated)).to(args.device)
            restored.load_state_dict(candidate.state_dict())
            restored_optimizer = torch.optim.Adam(optimizer_groups(restored, 1e-6, 1e-4))
            restored_optimizer.load_state_dict(state)
            assert set_training_stage(restored, 4, 3) == 'joint'
            set_training_stage(candidate, 4, 3)
            for model, opt in ((candidate, optimizer), (restored, restored_optimizer)):
                torch.manual_seed(456)
                opt.zero_grad()
                model(inp, coord, coord.new_tensor([2.5]), cell).square().mean().backward()
                opt.step()
            assert all(torch.allclose(p, q, atol=1e-6, rtol=0) for p, q in zip(
                candidate.parameters(), restored.parameters()))
            assert any(not torch.equal(original[name], p) for name, p in
                       candidate.named_parameters() if name in original)
            candidate.eval()
            with torch.no_grad():
                candidate.gen_feat(inp)
                full = candidate.query_rgb(coord, coord.new_tensor([2.5]), cell)
                chunks = torch.cat([candidate.query_rgb(
                    coord[:, i:i+7], coord.new_tensor([2.5]), cell[:, i:i+7],
                ) for i in range(0, coord.shape[1], 7)], dim=1)
            assert torch.allclose(full, chunks, atol=1e-6, rtol=0)
            try:
                initialize_from_baseline(candidate, path, 2)
            except ValueError:
                pass
            else:
                raise AssertionError('Wrong seed was accepted.')
    print('SCALE-GATED FREQUENCY MECHANISM TEST PASSED')


if __name__ == '__main__':
    main()
