from __future__ import annotations

import errno
from typing import Any, Dict, Optional


def safe_load_file(path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    A robust alternative to `safetensors.torch.load_file`.

    Why:
    - On some filesystems / mounts (common on HPC / DGX setups), mmap can fail and
      `safetensors.safe_open()` raises ENODEV (os error 19).
    - Falling back to loading from bytes avoids mmap entirely.
    """
    from safetensors.torch import load_file

    try:
        return load_file(path, device=device)
    except OSError as e:
        # ENODEV == 19 ("No such device"): often indicates the underlying FS doesn't
        # support mmap() for this file.
        if getattr(e, "errno", None) not in (errno.ENODEV, 19):
            raise

        # Fallback: load from raw bytes, which avoids mmap.
        from safetensors.torch import load

        with open(path, "rb") as f:
            data = f.read()

        # Newer versions accept `device=...`; older may not. Support both.
        try:
            return load(data, device=device)  # type: ignore[arg-type]
        except TypeError:
            state_dict = load(data)  # type: ignore[arg-type]
            if device and device != "cpu":
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
            return state_dict

