"""Check phase routing, parameter sharing and the split exposure schedule on CPU."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interaction"))
from train_interaction import build_model, load_runtime
from prepare_predictor_plans import split_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="append", nargs=2, metavar=("CONFIG", "CHECKPOINT"))
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    for config, checkpoint in args.weights or []:
        cfg = OmegaConf.load(config)
        runtime, _, module = load_runtime(cfg)
        model = build_model(cfg, runtime, module)
        load_model(model, checkpoint, strict=True, device="cpu")
        print(f"Strict checkpoint load passed: {Path(config).name} ({sum(p.numel() for p in model.parameters()):,} parameters)")
        del model
    for variant in ("pavg", "without_cpi", "physctrl", "unified"):
        cfg = OmegaConf.load(ROOT / "configs/predictors" / f"{variant}.yaml")
        cfg.pc_size, cfg.n_training_frames = 16, 2
        cfg.model_config.n_layers = 1
        cfg.model_config.latent_dim = 64
        cfg.model_config.dit_cross_attention_heads = 1
        runtime, _, module = load_runtime(cfg)
        model = build_model(cfg, runtime, module).eval()
        p = cfg.pc_size
        noisy = torch.randn(1, 2, 2, p, 3)
        points = torch.randn(1, 2, p, 3)
        data = dict(init_vel=torch.zeros_like(points), force=torch.ones(1, 2, 3),
                    drag_point=torch.zeros(1, 2, 3), drag_mask=torch.ones(1, 2, p, 1),
                    physics_E=torch.ones(1, 2, 1), physics_nu=torch.ones(1, 2, 1),
                    physics_mat_type=torch.tensor([[4, 0]]), physics_floor_height=torch.zeros(1, 2, 1),
                    physics_gravity=torch.zeros(1, 2, dtype=torch.long), physics_coeff=torch.ones(1, 2, 1),
                    physics_volume=torch.ones(1, 2, p, 1))
        assert model.cond_seq_length == (7 if variant in ("physctrl", "unified") else 4)
        assert model.object_cond_seq_length == 7
        assert (model.dual_dit.gas_dit is model.dual_dit.object_dit) == (variant in ("physctrl", "unified"))
        assert model.cross_attention_enabled == (variant == "pavg")
        with torch.inference_mode():
            def call(gas, solid):
                return model(noisy, torch.tensor([3]), points, **data,
                             phase_presence=torch.tensor([[gas, solid]]))
            pair = call(True, True)
            assert pair.shape == noisy.shape and torch.isfinite(pair).all()
            for slot, single in enumerate((call(True, False), call(False, True))):
                assert not torch.count_nonzero(single[:, :, 1 - slot])
                if variant != "pavg":
                    torch.testing.assert_close(single[:, :, slot], pair[:, :, slot])
        print(f"Passed reduced CPU forward: {variant}, gas / solid / paired")
    plan, counts, weights = split_plan(np.array([0, 1, 2, 2]), np.array([0, 1, 2]))
    assert sorted(plan[plan >= 0]) == [0, 3, 4, 4, 5, 5]
    np.testing.assert_array_equal(counts, [3, 3])
    np.testing.assert_array_equal(weights, [1, 1])
    print("Passed independent phase exposure and weight checks")


if __name__ == "__main__":
    main()
