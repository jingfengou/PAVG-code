"""Opt-in execution changes that preserve the model and training objective."""

from functools import partial

import torch


def selective_checkpoint_context(policy):
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    operations = [torch.ops.aten._scaled_dot_product_flash_attention.default]
    if policy == "attention_matmul":
        operations += [torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
                       torch.ops.aten.bmm.default]
    elif policy != "attention":
        raise ValueError(f"Unknown selective checkpoint policy: {policy}")
    return partial(create_selective_checkpoint_contexts, operations)


def aggregate_nonfinite(named_tensors, limit=8, implementation="elementwise"):
    entries = [(name, tensor) for name, tensor in named_tensors if tensor is not None]
    if not entries:
        return []
    tensors = [tensor for _, tensor in entries]
    if implementation == "foreach":
        maxima = torch._foreach_norm([tensor for _, tensor in entries], float("inf"))
        flags = torch.isfinite(torch.stack(maxima))
    else:
        flags = torch.stack([torch.isfinite(tensor).all() for _, tensor in entries])
    if bool(flags.all()):
        return []
    return [name for (name, _), finite in zip(entries, flags.cpu().tolist())
            if not finite][:limit]


def configure_model_performance(model, cfg):
    torch.use_deterministic_algorithms(bool(cfg.get("deterministic_algorithms", False)))
    if cfg.get("fused_operator_mode", "none") != "none":
        raise ValueError("Use fused_operator_mode=none with the portable PyTorch backend")
    if hasattr(model, "solid"):
        trunks = (model.solid.dit,)
        if cfg.get("checkpoint_layers") is not None:
            model.solid.dit.checkpoint_layers = frozenset(int(i) for i in cfg.checkpoint_layers)
    else:
        trunks = (model.dual_dit.gas_dit, model.dual_dit.object_dit)
        selected = cfg.get("checkpoint_layers")
        if selected is not None:
            model.dual_dit.checkpoint_layers = frozenset(int(index) for index in selected)
        single_phase = cfg.get("single_phase_checkpoint_layers")
        if single_phase is not None:
            model.dual_dit.single_phase_checkpoint_layers = frozenset(int(index) for index in single_phase)
        packed = bool(cfg.get("pack_independent_phases", False))
        if packed:
            if not (model.dual_dit.joint_point_attention
                    and model.dual_dit.independent_phase_forward):
                raise ValueError("Packed execution requires independent shared-trunk phases")
            if any(isinstance(module, torch.nn.Dropout) and module.p
                   for module in model.dual_dit.gas_dit.modules()):
                raise ValueError("Packed execution requires zero trunk dropout")
            if getattr(model.dual_dit.gas_dit.class_embedder, "dropout_prob", 0):
                raise ValueError("Packed execution requires zero material dropout")
        model.dual_dit.pack_independent_phases = packed
    policy = cfg.get("selective_checkpoint_policy", "none")
    if policy != "none":
        context = selective_checkpoint_context(policy)
        if hasattr(model, "solid"):
            model.solid.dit.selective_checkpoint_context_fn = context
        else:
            if not model.dual_dit.joint_point_attention:
                raise ValueError("Selective checkpoint qualification covers joint unified models")
            model.dual_dit.selective_checkpoint_context_fn = context
            model.dual_dit.selective_checkpoint_single_phase_only = bool(
                cfg.get("selective_checkpoint_single_phase_only", True))
    if bool(cfg.get("compile_transformer_blocks", False)):
        seen = set()
        for trunk in trunks:
            for block in trunk.transformer_blocks:
                if id(block) in seen:
                    continue
                seen.add(id(block))
                # Compile forward only, retaining original checkpoint parameter names.
                block.forward = torch.compile(block.forward, dynamic=True, fullgraph=False)


def loader_performance_kwargs(cfg):
    if int(cfg.num_workers) <= 0:
        return {}
    return {
        "prefetch_factor": int(cfg.get("prefetch_factor", 2)),
        "persistent_workers": bool(cfg.get("persistent_workers", False)),
    }


def configure_restored_optimizer(optimizer, cfg):
    if not bool(cfg.get("fused_adamw", False)):
        return
    # load_state_dict restores the old param-group flags and CPU step counters.
    for group in optimizer.param_groups:
        group["fused"], group["foreach"] = True, None
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if isinstance(state.get("step"), torch.Tensor):
                state["step"] = state["step"].to(parameter.device)
