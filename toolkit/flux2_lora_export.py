"""
Flux2 / Flux2-klein LoRA 导出工具：

ai-toolkit 训练时保存的 Flux2 LoRA key 通常是（PEFT lora_A/lora_B，但模块路径是 comfy 风格）：
  - transformer.single_blocks.N.linear1/linear2...
  - transformer.double_blocks.N.(img_attn|txt_attn|img_mlp|txt_mlp)...
并且很多训练是 partial（例如 single_blocks 只有 0~23），这会导致 diffusers 内置的
`_convert_non_diffusers_flux2_lora_to_diffusers()`（按固定 48 层 pop）直接 KeyError。

这里提供一个“按实际存在的 key”转换成最终 diffusers/PEFT 可直接加载的格式：
  - transformer.single_transformer_blocks.N.attn.to_qkv_mlp_proj.*.lora_A/B.weight
  - transformer.single_transformer_blocks.N.attn.to_out.*.lora_A/B.weight
  - transformer.transformer_blocks.N.attn.(to_q/to_k/to_v|add_q_proj/add_k_proj/add_v_proj).*lora_A/B.weight
  - transformer.transformer_blocks.N.attn.(to_out.0|to_add_out).*lora_A/B.weight
  - transformer.transformer_blocks.N.(ff|ff_context).(linear_in|linear_out).*lora_A/B.weight
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional

import torch


def _strip_known_prefix(k: str) -> str:
    if k.startswith("diffusion_model."):
        return k[len("diffusion_model.") :]
    if k.startswith("transformer."):
        return k[len("transformer.") :]
    return k


def _ensure_transformer_prefix(k: str) -> str:
    return k if k.startswith("transformer.") else f"transformer.{k}"


def _clone_for_safetensors(x: torch.Tensor) -> torch.Tensor:
    # safetensors 不允许多个 key 共享同一底层存储（shared storage / view）
    return x.contiguous().clone()


def convert_flux2_lora_to_diffusers_format(
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """
    将 ai-toolkit / comfy 风格的 Flux2 LoRA state_dict 转成最终 diffusers/PEFT 可直接加载的格式。
    - 只处理包含 `.lora_A.weight` / `.lora_B.weight` 的 key
    - 允许 partial LoRA（缺层时不会 KeyError）
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    normalized = {_strip_known_prefix(k): v for k, v in state_dict.items()}

    # 1) single blocks: single_blocks.N.(linear1|linear2).lora_[AB].weight
    for k, v in list(normalized.items()):
        if not k.startswith("single_blocks."):
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        parts = k.split(".")
        if len(parts) < 5:
            if strict:
                raise ValueError(f"Unrecognized single_blocks key: {k}")
            continue
        idx = parts[1]
        sub = ".".join(parts[2:])  # linear1.lora_A.weight

        if sub.startswith("linear1."):
            suffix = sub.replace("linear1.", "attn.to_qkv_mlp_proj.", 1)
            out[_ensure_transformer_prefix(f"single_transformer_blocks.{idx}.{suffix}")] = _clone_for_safetensors(v)
        elif sub.startswith("linear2."):
            suffix = sub.replace("linear2.", "attn.to_out.", 1)
            out[_ensure_transformer_prefix(f"single_transformer_blocks.{idx}.{suffix}")] = _clone_for_safetensors(v)
        else:
            if strict:
                raise ValueError(f"Unrecognized single_blocks subkey: {k}")

    # 2) double blocks: fused qkv
    for k, v in list(normalized.items()):
        if not k.startswith("double_blocks."):
            continue
        if ".qkv." not in k:
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue

        parts = k.split(".")
        if len(parts) < 6:
            if strict:
                raise ValueError(f"Unrecognized double_blocks qkv key: {k}")
            continue

        dl = parts[1]
        attn_type = parts[2]  # img_attn / txt_attn
        lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"

        if attn_type not in ("img_attn", "txt_attn"):
            if strict:
                raise ValueError(f"Unrecognized attn_type in key: {k}")
            continue

        attn_prefix = f"transformer_blocks.{dl}.attn"

        if lora_key == "lora_A":
            proj_keys = ["to_q", "to_k", "to_v"] if attn_type == "img_attn" else ["add_q_proj", "add_k_proj", "add_v_proj"]
            for pk in proj_keys:
                out[_ensure_transformer_prefix(f"{attn_prefix}.{pk}.lora_A.weight")] = _clone_for_safetensors(v)
        else:
            # B: fused qkv 必须拆开
            try:
                q, k_, v_ = torch.chunk(v, 3, dim=0)
            except Exception:
                if strict:
                    raise
                continue
            q = _clone_for_safetensors(q)
            k_ = _clone_for_safetensors(k_)
            v_ = _clone_for_safetensors(v_)
            if attn_type == "img_attn":
                out[_ensure_transformer_prefix(f"{attn_prefix}.to_q.lora_B.weight")] = q
                out[_ensure_transformer_prefix(f"{attn_prefix}.to_k.lora_B.weight")] = k_
                out[_ensure_transformer_prefix(f"{attn_prefix}.to_v.lora_B.weight")] = v_
            else:
                out[_ensure_transformer_prefix(f"{attn_prefix}.add_q_proj.lora_B.weight")] = q
                out[_ensure_transformer_prefix(f"{attn_prefix}.add_k_proj.lora_B.weight")] = k_
                out[_ensure_transformer_prefix(f"{attn_prefix}.add_v_proj.lora_B.weight")] = v_

    # 3) double blocks: proj
    proj_mappings = [
        ("img_attn.proj", "attn.to_out.0"),
        ("txt_attn.proj", "attn.to_add_out"),
    ]
    for k, v in list(normalized.items()):
        if not k.startswith("double_blocks."):
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"
        for org_proj, diff_proj in proj_mappings:
            if f".{org_proj}." not in k:
                continue
            dl = k.split(".", 3)[1]
            out[_ensure_transformer_prefix(f"transformer_blocks.{dl}.{diff_proj}.{lora_key}.weight")] = _clone_for_safetensors(v)

    # 4) double blocks: mlp
    mlp_mappings = [
        ("img_mlp.0", "ff.linear_in"),
        ("img_mlp.2", "ff.linear_out"),
        ("txt_mlp.0", "ff_context.linear_in"),
        ("txt_mlp.2", "ff_context.linear_out"),
    ]
    for k, v in list(normalized.items()):
        if not k.startswith("double_blocks."):
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"
        for org_mlp, diff_mlp in mlp_mappings:
            if f".{org_mlp}." not in k:
                continue
            dl = k.split(".", 3)[1]
            out[_ensure_transformer_prefix(f"transformer_blocks.{dl}.{diff_mlp}.{lora_key}.weight")] = _clone_for_safetensors(v)

    # 5) 已经是最终结构的 key（例如 assistant LoRA）直接保留（仅 clone + 补 transformer. 前缀）
    for k, v in list(normalized.items()):
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        if k.startswith("single_transformer_blocks.") or k.startswith("transformer_blocks."):
            out[_ensure_transformer_prefix(k)] = _clone_for_safetensors(v)

    if len(out) == 0:
        raise ValueError("No convertible Flux2 LoRA keys found (expected *.lora_A/B.weight).")

    # 最终自检：diffusers 会要求所有 key 都包含 'lora' 子串
    bad = [k for k in out.keys() if "lora" not in k]
    if bad:
        raise RuntimeError(f"Converted state_dict contains non-lora keys: {bad[:20]}")

    return out


def convert_flux2_lora_from_diffusers_format(
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """
    逆转换：把最终 diffusers/PEFT 的 Flux2 LoRA（single_transformer_blocks/transformer_blocks）
    转回 ai-toolkit 训练时使用的 comfy 风格（single_blocks/double_blocks）。

    这样即便断点续训误加载了 *_diffusers.safetensors，也能正确恢复训练。
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    normalized = {_strip_known_prefix(k): v for k, v in state_dict.items()}

    # 1) single: single_transformer_blocks.N.attn.to_qkv_mlp_proj -> single_blocks.N.linear1
    #    single_transformer_blocks.N.attn.to_out -> single_blocks.N.linear2
    for k, v in list(normalized.items()):
        if not k.startswith("single_transformer_blocks."):
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        parts = k.split(".")
        if len(parts) < 6:
            if strict:
                raise ValueError(f"Unrecognized single_transformer_blocks key: {k}")
            continue
        idx = parts[1]
        if ".attn.to_qkv_mlp_proj." in k:
            new_k = k.replace(f"single_transformer_blocks.{idx}.attn.to_qkv_mlp_proj.", f"single_blocks.{idx}.linear1.")
            out[_ensure_transformer_prefix(new_k)] = _clone_for_safetensors(v)
        elif ".attn.to_out." in k:
            new_k = k.replace(f"single_transformer_blocks.{idx}.attn.to_out.", f"single_blocks.{idx}.linear2.")
            out[_ensure_transformer_prefix(new_k)] = _clone_for_safetensors(v)
        else:
            if strict:
                raise ValueError(f"Unsupported single_transformer_blocks mapping for key: {k}")

    # 2) double: proj / mlp
    for k, v in list(normalized.items()):
        if not k.startswith("transformer_blocks."):
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue

        dl = k.split(".", 2)[1]
        lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"

        # proj
        if f"transformer_blocks.{dl}.attn.to_out.0.{lora_key}.weight" in k:
            out[_ensure_transformer_prefix(f"double_blocks.{dl}.img_attn.proj.{lora_key}.weight")] = _clone_for_safetensors(v)
            continue
        if f"transformer_blocks.{dl}.attn.to_add_out.{lora_key}.weight" in k:
            out[_ensure_transformer_prefix(f"double_blocks.{dl}.txt_attn.proj.{lora_key}.weight")] = _clone_for_safetensors(v)
            continue

        # mlp
        mlp_back = [
            ("ff.linear_in", "img_mlp.0"),
            ("ff.linear_out", "img_mlp.2"),
            ("ff_context.linear_in", "txt_mlp.0"),
            ("ff_context.linear_out", "txt_mlp.2"),
        ]
        matched = False
        for diff_mlp, org_mlp in mlp_back:
            if f"transformer_blocks.{dl}.{diff_mlp}.{lora_key}.weight" in k:
                out[_ensure_transformer_prefix(f"double_blocks.{dl}.{org_mlp}.{lora_key}.weight")] = _clone_for_safetensors(v)
                matched = True
                break
        if matched:
            continue

    # 3) double: qkv（需要把 q/k/v 合成 fused qkv）
    # 收集后再合成，避免顺序依赖
    qkv_bins: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    # key: (dl, attn_type, lora_key) -> {"q": t, "k": t, "v": t}
    for k, v in list(normalized.items()):
        if not k.startswith("transformer_blocks."):
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        dl = k.split(".", 2)[1]
        lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"

        # img_attn: to_q/to_k/to_v
        for proj, slot in (("to_q", "q"), ("to_k", "k"), ("to_v", "v")):
            if f"transformer_blocks.{dl}.attn.{proj}.{lora_key}.weight" == k:
                bin_key = (dl, "img_attn", lora_key)
                qkv_bins.setdefault(bin_key, {})[slot] = v
        # txt_attn: add_q_proj/add_k_proj/add_v_proj
        for proj, slot in (("add_q_proj", "q"), ("add_k_proj", "k"), ("add_v_proj", "v")):
            if f"transformer_blocks.{dl}.attn.{proj}.{lora_key}.weight" == k:
                bin_key = (dl, "txt_attn", lora_key)
                qkv_bins.setdefault(bin_key, {})[slot] = v

    for (dl, attn_type, lora_key), slots in qkv_bins.items():
        if lora_key == "lora_A":
            # A 在我们的正向转换里是复制的，逆向取 q 即可
            if "q" not in slots:
                if strict:
                    raise ValueError(f"Missing q for A qkv at block {dl} {attn_type}")
                continue
            fused = slots["q"]
        else:
            if not all(s in slots for s in ("q", "k", "v")):
                if strict:
                    raise ValueError(f"Missing q/k/v for B qkv at block {dl} {attn_type}")
                continue
            fused = torch.cat([slots["q"], slots["k"], slots["v"]], dim=0)
        out[_ensure_transformer_prefix(f"double_blocks.{dl}.{attn_type}.qkv.{lora_key}.weight")] = _clone_for_safetensors(fused)

    if len(out) == 0:
        raise ValueError("No convertible diffusers Flux2 LoRA keys found.")
    return out

