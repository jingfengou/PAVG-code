"""Continue a frozen exposure schedule after changing the global batch size."""

import copy
import hashlib
import math
from pathlib import Path

from torch.optim.lr_scheduler import LambdaLR


def exposure_cursor(step, global_batch, transition=None):
    if not transition:
        return int(step) * int(global_batch)
    pivot = int(transition["step"])
    previous = int(transition["global_batch"])
    return min(int(step), pivot) * previous + max(int(step) - pivot, 0) * int(global_batch)


def transition_scheduler(optimizer, cfg, world_size, split_batches=False):
    transition = cfg.batch_transition
    multiplier = 1 if split_batches else int(world_size)
    previous_batch = int(transition.global_batch)
    new_batch = int(cfg.exposure_plan.global_batch)
    warmup_samples = int(cfg.lr_warmup_steps) * previous_batch
    total_samples = int(transition.max_train_steps) * previous_batch

    def factor(scheduler_step):
        # Keep fractional scheduler substeps; Accelerate advances once per rank.
        optimizer_step = scheduler_step / multiplier
        pivot = int(transition.step)
        samples = (min(optimizer_step, pivot) * previous_batch
                   + max(optimizer_step - pivot, 0) * new_batch)
        if samples < warmup_samples:
            return samples / max(1, warmup_samples)
        progress = (samples - warmup_samples) / max(1, total_samples - warmup_samples)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, factor)


def validate_resume_contract(previous, current, cfg, resume_path):
    if previous == current:
        return
    transition = cfg.get("batch_transition")
    if not transition:
        raise ValueError("Checkpoint training contract differs")
    if Path(resume_path).resolve() != Path(str(transition.checkpoint)).resolve():
        raise ValueError("Batch transition must start from its audited source checkpoint")
    if int(Path(resume_path).name.rsplit("-", 1)[1]) != int(transition.step):
        raise ValueError("Batch transition checkpoint step differs")
    contract_path = Path(resume_path).parent / "training_contract.json"
    if transition.get("contract_sha256") and hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest() != transition.contract_sha256:
        raise ValueError("Source training contract hash differs")
    expected = copy.deepcopy(current)
    expected.pop("batch_transition")
    expected["max_optimizer_steps"] = int(transition.max_train_steps)
    expected["exposure_plan"]["global_batch"] = int(transition.global_batch)
    if previous != expected:
        raise ValueError("Batch transition changed fields beyond its authorized batch and horizon")
