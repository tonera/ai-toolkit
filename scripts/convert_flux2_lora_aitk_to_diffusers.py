"""
把 ai-toolkit 训练导出的 Flux2/Flux2-klein LoRA 做“确定性的结构清洗”，让 diffusers 更容易加载：
- 先打印源 LoRA 的 key 结构概览（层覆盖范围、是否是 partial LoRA、key 命名风格）
- 再把 ai-toolkit / comfy 风格的 key **转换成最终 diffusers/PEFT 可直接加载的格式**：
  - 输出 key 统一为 `transformer.*.lora_A.weight` / `transformer.*.lora_B.weight`
  - 不再依赖 diffusers 内置 `_convert_non_diffusers_flux2_lora_to_diffusers()`（它假设 LoRA 覆盖固定层数，partial LoRA 会 KeyError）

这一步的目标是：把 ai-toolkit 的 LoRA key 变成 diffusers 的 flux2 “非 diffusers LoRA 转换器”
更容易识别的输入格式（`single_blocks.*` / `double_blocks.*`）。

为什么需要：
- ai-toolkit 的 Flux2 保存 LoRA 时会把 key 从 `transformer.*` 改成 `diffusion_model.*`
  （见 `extensions_built_in/diffusion_models/flux2/flux2_model.py:convert_lora_weights_before_save`）
- diffusers 内置的 ai-toolkit Flux2 转换器会按“固定层数”去 `.pop()` 预期 key：
  对于 **partial LoRA / 层数不一致** 的情况会直接 `KeyError`（你遇到的就是这个）

用法：
  python scripts/convert_flux2_lora_aitk_to_diffusers.py \
    --in  /home/tonera/project/ai-toolkit/output/sakimi_flux2_klein_9b_style_lora_v2_stable/sakimi_flux2_klein_9b_style_lora_v2_stable.safetensors \
    --out /home/tonera/project/ai-toolkit/output/sakimi_flux2_klein_9b_style_lora_v2_stable/sakimi_flux2_klein_9b_style_lora_v2_stable_diffusers.safetensors
"""

import argparse
from collections import OrderedDict

import re


_RE_SINGLE_TR = re.compile(r"^single_transformer_blocks\.(\d+)\.")
_RE_DOUBLE_TR = re.compile(r"^transformer_blocks\.(\d+)\.")
_RE_SINGLE_BLOCKS = re.compile(r"^single_blocks\.(\d+)\.")
_RE_DOUBLE_BLOCKS = re.compile(r"^double_blocks\.(\d+)\.")


def _analyze_keys(keys: list[str]) -> dict:
    base_keys = [_strip_known_prefix(k) for k in keys]
    info = {
        "count": len(keys),
        "has_diffusion_model_prefix": any(k.startswith("diffusion_model.") for k in keys),
        "has_transformer_prefix": any(k.startswith("transformer.") for k in keys),
        "has_single_blocks": any(k.startswith("single_blocks.") for k in base_keys),
        "has_double_blocks": any(k.startswith("double_blocks.") for k in base_keys),
        "has_single_transformer_blocks": any(k.startswith("single_transformer_blocks.") for k in base_keys),
        "has_transformer_blocks": any(k.startswith("transformer_blocks.") for k in base_keys),
        "has_lora_A": any(".lora_A.weight" in k for k in base_keys),
        "has_lora_B": any(".lora_B.weight" in k for k in base_keys),
        "has_lora_down": any(".lora_down.weight" in k for k in base_keys),
        "has_lora_up": any(".lora_up.weight" in k for k in base_keys),
        "has_linear1": any(".linear1." in k for k in base_keys),
        "has_attn_to_q": any(".attn.to_q." in k for k in base_keys),
    }
    # block index ranges
    def max_idx(prefix: str) -> int | None:
        idxs = []
        for k in base_keys:
            if k.startswith(prefix):
                try:
                    idxs.append(int(k.split(".")[1]))
                except Exception:
                    pass
        return max(idxs) if idxs else None

    info["max_single_blocks_idx"] = max_idx("single_blocks.")
    info["max_double_blocks_idx"] = max_idx("double_blocks.")
    info["max_single_transformer_blocks_idx"] = max_idx("single_transformer_blocks.")
    info["max_transformer_blocks_idx"] = max_idx("transformer_blocks.")
    return info


def _strip_known_prefix(k: str) -> str:
    if k.startswith("diffusion_model."):
        return k[len("diffusion_model.") :]
    if k.startswith("transformer."):
        return k[len("transformer.") :]
    return k


def inspect_lora_state_dict(state_dict: dict, *, max_examples: int = 30) -> None:
    keys = list(state_dict.keys())
    info = _analyze_keys(keys)
    print("[inspect] summary:", info)

    # 粗略统计 top-level 前缀
    buckets = {}
    for k in keys:
        base = _strip_known_prefix(k)
        top = base.split(".", 1)[0] if "." in base else base
        buckets[top] = buckets.get(top, 0) + 1
    buckets_sorted = sorted(buckets.items(), key=lambda x: (-x[1], x[0]))
    print("[inspect] top-level buckets (top 20):", buckets_sorted[:20])

    # 打印一些代表性 key，方便你写映射规则
    print(f"[inspect] example keys (first {max_examples}):")
    for k in keys[:max_examples]:
        print("  -", k)

    # 统计 single/double block index 覆盖
    single_idxs = set()
    double_idxs = set()
    for k in keys:
        base = _strip_known_prefix(k)
        m = _RE_SINGLE_BLOCKS.match(base) or _RE_SINGLE_TR.match(base)
        if m:
            single_idxs.add(int(m.group(1)))
        m = _RE_DOUBLE_BLOCKS.match(base) or _RE_DOUBLE_TR.match(base)
        if m:
            double_idxs.add(int(m.group(1)))

    if single_idxs:
        print("[inspect] single idx range:", (min(single_idxs), max(single_idxs)), "count:", len(single_idxs))
    if double_idxs:
        print("[inspect] double idx range:", (min(double_idxs), max(double_idxs)), "count:", len(double_idxs))


def _ensure_transformer_prefix(k: str) -> str:
    return k if k.startswith("transformer.") else f"transformer.{k}"

def _clone_for_safetensors(x):
    """
    safetensors 不允许多个 key 共享同一底层存储（shared storage / view）。
    这里把 tensor 变成独立 contiguous 拷贝；非 tensor 原样返回。
    """
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            # contiguous().clone() 可以同时打断 view/共享存储
            return x.contiguous().clone()
    except Exception:
        # 没有 torch 的环境下，这个函数不会被用于需要 tensor 操作的分支
        pass
    return x


def convert_keys_to_final_diffusers_format(state_dict: dict, *, strict: bool = False) -> "OrderedDict[str, object]":
    """
    把输入 LoRA（ai-toolkit / comfy 风格）转换为最终 diffusers/PEFT 可直接 load 的 state_dict：
    - key 必须包含 `transformer.` 前缀（PEFT loader 会按 prefix 过滤）
    - key 必须包含 `lora_A/lora_B`（否则会被当作非 LoRA 或走其它转换）

    strict=False 时，遇到不认识/不完整的 key 会跳过并打印 warning（适配 partial LoRA）。
    """
    # torch 仅在处理 double_blocks 的 fused qkv（需要 chunk）时才需要
    torch = None

    out = OrderedDict()

    # 1) 先把 key 归一化到“无前缀”的 body
    normalized = { _strip_known_prefix(k): v for k, v in state_dict.items() }

    # 2) single blocks: single_blocks.N.(linear1|linear2).lora_[AB].weight
    for k, v in list(normalized.items()):
        if not k.startswith("single_blocks."):
            continue
        # e.g. single_blocks.24.linear1.lora_A.weight
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        parts = k.split(".")
        if len(parts) < 5:
            if strict:
                raise ValueError(f"Unrecognized single_blocks key: {k}")
            print("[warn] skip key:", k)
            continue
        idx = parts[1]
        sub = ".".join(parts[2:])  # linear1.lora_A.weight
        if sub.startswith("linear1."):
            # -> single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.lora_A/B.weight
            suffix = sub.replace("linear1.", "attn.to_qkv_mlp_proj.", 1)
            out[_ensure_transformer_prefix(f"single_transformer_blocks.{idx}.{suffix}")] = v
        elif sub.startswith("linear2."):
            # -> single_transformer_blocks.{i}.attn.to_out.lora_A/B.weight
            suffix = sub.replace("linear2.", "attn.to_out.", 1)
            out[_ensure_transformer_prefix(f"single_transformer_blocks.{idx}.{suffix}")] = v
        else:
            # 未知 single_blocks 子结构：保守跳过/或原样输出到 transformer.single_blocks（但 Flux2 模型里没有这个模块名）
            if strict:
                raise ValueError(f"Unrecognized single_blocks subkey: {k}")
            print("[warn] skip single_blocks key (unknown sub-structure):", k)

    # 3) double blocks: comfy 风格 double_blocks.{dl}.(img_attn|txt_attn|img_mlp|txt_mlp...).*.lora_[AB].weight
    # 参考 diffusers 的映射逻辑（但不按固定层数 pop，按实际存在的 key 转）
    # 3.1 qkv fused
    for k, v in list(normalized.items()):
        if not k.startswith("double_blocks."):
            continue
        if ".qkv." not in k:
            continue
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue

        # e.g. double_blocks.0.img_attn.qkv.lora_B.weight
        parts = k.split(".")
        if len(parts) < 6:
            if strict:
                raise ValueError(f"Unrecognized double_blocks qkv key: {k}")
            print("[warn] skip key:", k)
            continue
        dl = parts[1]
        attn_type = parts[2]  # img_attn / txt_attn
        lora_key = parts[4]   # lora_A / lora_B
        if attn_type not in ("img_attn", "txt_attn"):
            if strict:
                raise ValueError(f"Unrecognized attn_type in key: {k}")
            print("[warn] skip key:", k)
            continue

        attn_prefix = f"transformer_blocks.{dl}.attn"
        if lora_key == "lora_A":
            # A: 直接复用同一个 A 到 q/k/v
            proj_keys = ["to_q", "to_k", "to_v"] if attn_type == "img_attn" else ["add_q_proj", "add_k_proj", "add_v_proj"]
            for pk in proj_keys:
                # 重要：不能复用同一 tensor 对象，否则 safetensors 会报 shared memory
                out[_ensure_transformer_prefix(f"{attn_prefix}.{pk}.lora_A.weight")] = _clone_for_safetensors(v)
        else:
            # B: 按 dim=0 拆成 q/k/v
            if torch is None:
                try:
                    import torch as _torch  # type: ignore
                    torch = _torch
                except Exception as e:
                    if strict:
                        raise
                    print("[warn] torch not available, skip fused qkv conversion for key:", k, "error:", repr(e))
                    continue
            try:
                sample_q, sample_k, sample_v = torch.chunk(v, 3, dim=0)
            except Exception as e:
                if strict:
                    raise
                print("[warn] cannot chunk fused qkv for key:", k, "error:", repr(e))
                continue
            # 重要：chunk 出来的是 view，依然共享存储；保存前必须 clone
            sample_q = _clone_for_safetensors(sample_q)
            sample_k = _clone_for_safetensors(sample_k)
            sample_v = _clone_for_safetensors(sample_v)
            if attn_type == "img_attn":
                out[_ensure_transformer_prefix(f"{attn_prefix}.to_q.lora_B.weight")] = sample_q
                out[_ensure_transformer_prefix(f"{attn_prefix}.to_k.lora_B.weight")] = sample_k
                out[_ensure_transformer_prefix(f"{attn_prefix}.to_v.lora_B.weight")] = sample_v
            else:
                out[_ensure_transformer_prefix(f"{attn_prefix}.add_q_proj.lora_B.weight")] = sample_q
                out[_ensure_transformer_prefix(f"{attn_prefix}.add_k_proj.lora_B.weight")] = sample_k
                out[_ensure_transformer_prefix(f"{attn_prefix}.add_v_proj.lora_B.weight")] = sample_v

    # 3.2 proj mappings
    proj_mappings = [
        ("img_attn.proj", "attn.to_out.0"),
        ("txt_attn.proj", "attn.to_add_out"),
    ]
    for org_proj, diff_proj in proj_mappings:
        for k, v in list(normalized.items()):
            if not k.startswith("double_blocks."):
                continue
            if f".{org_proj}." not in k:
                continue
            if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
                continue
            parts = k.split(".")
            dl = parts[1]
            # 注意：double_blocks.0.img_attn.proj.lora_A.weight 的 split[-3] 是 "proj"（会生成错误 key）
            # 这里必须稳定地取 lora_A / lora_B
            lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"
            out[_ensure_transformer_prefix(f"transformer_blocks.{dl}.{diff_proj}.{lora_key}.weight")] = v

    # 3.3 mlp mappings
    mlp_mappings = [
        ("img_mlp.0", "ff.linear_in"),
        ("img_mlp.2", "ff.linear_out"),
        ("txt_mlp.0", "ff_context.linear_in"),
        ("txt_mlp.2", "ff_context.linear_out"),
    ]
    for org_mlp, diff_mlp in mlp_mappings:
        for k, v in list(normalized.items()):
            if not k.startswith("double_blocks."):
                continue
            if f".{org_mlp}." not in k:
                continue
            if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
                continue
            parts = k.split(".")
            dl = parts[1]
            # 注意：double_blocks.0.img_mlp.0.lora_A.weight 的 split[-3] 是 "0"（会生成错误 key）
            lora_key = "lora_A" if k.endswith(".lora_A.weight") else "lora_B"
            out[_ensure_transformer_prefix(f"transformer_blocks.{dl}.{diff_mlp}.{lora_key}.weight")] = v

    # 4) 已经是 diffusers 目标结构的（例如 single_transformer_blocks/transformer_blocks 开头）就原样加 transformer. 前缀
    for k, v in list(normalized.items()):
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        if k.startswith("single_transformer_blocks.") or k.startswith("transformer_blocks."):
            out[_ensure_transformer_prefix(k)] = v

    if len(out) == 0:
        raise ValueError(
            "转换结果为空：没有识别到任何可转换的 LoRA key。请先用 --inspect 查看源文件 key 结构。"
        )
    return out


def main():
    # 延迟导入：避免在仅做 key 转换/单测导入该模块时强依赖 torch
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="input .safetensors lora path")
    p.add_argument("--out", dest="output_path", required=True, help="output .safetensors path")
    p.add_argument("--inspect", action="store_true", help="only print key structure and exit (no conversion)")
    p.add_argument("--strict", action="store_true", help="strict mode: unknown/malformed keys raise error")
    args = p.parse_args()

    sd = load_file(args.input_path)
    keys = list(sd.keys())
    info_before = _analyze_keys(keys)

    print("[inspect] before:", info_before)
    inspect_lora_state_dict(sd)

    if args.inspect:
        return

    converted = convert_keys_to_final_diffusers_format(sd, strict=args.strict)
    # 强制打断任何可能的 shared storage / view，避免 safetensors 保存报错
    converted = OrderedDict((k, _clone_for_safetensors(v)) for k, v in converted.items())

    # 自检：最终输出必须是 transformer.*，不能再包含 diffusion_model.*
    bad = [k for k in converted.keys() if k.startswith("diffusion_model.")]
    if bad:
        raise RuntimeError(f"[fatal] output contains diffusion_model.* keys (should be transformer.* only): {bad[:5]}")
    if not any(k.startswith("transformer.") for k in converted.keys()):
        raise RuntimeError("[fatal] output contains no transformer.* keys; conversion likely failed.")

    info_after = _analyze_keys(list(converted.keys()))

    print("[inspect] after: ", info_after)

    meta = OrderedDict()
    meta["aitk.converted_from"] = args.input_path
    meta["aitk.converter"] = "convert_flux2_lora_aitk_to_diffusers.py"
    meta["aitk.converter_mode"] = "final_diffusers_format"
    save_file(converted, args.output_path, metadata=meta)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()

