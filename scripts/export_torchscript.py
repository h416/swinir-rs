#!/usr/bin/env python3
"""Export SwinIR models to TorchScript format.

Usage: python scripts/export_torchscript.py
Output: weights/swinir_real_x{2,4}_traced.pt

Prerequisites:
  - pip install torch timm
  - Download pretrained weights to weights/ directory (see README.md)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from network_swinir import SwinIR

PROJECT_ROOT = Path(__file__).parent.parent

SWINIR_WEIGHTS = {
    2: PROJECT_ROOT / "weights" / "swinir_real_x2_gan.pth",
    4: PROJECT_ROOT / "weights" / "swinir_real_x4_gan.pth",
}

SWINIR_MODEL_PARAMS = dict(
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="nearest+conv",
    resi_connection="1conv",
)

OUTPUT_DIR = PROJECT_ROOT / "weights"


def export(scale: int) -> None:
    weight_path = SWINIR_WEIGHTS[scale]
    if not weight_path.exists():
        print(f"Weight file not found: {weight_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Exporting x{scale} ===")
    model = SwinIR(upscale=scale, **SWINIR_MODEL_PARAMS)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=False)
    model.eval()
    model = model.float()

    # Trace on MPS to embed device info in the graph
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  Device: {device}")
    model = model.to(device)

    # Trace with 256x256 dummy input (triggers dynamic mask calculation path)
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, check_trace=False)

    # Verify: compare PyTorch vs traced output
    test_input = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        out_orig = model(test_input)
        out_traced = traced(test_input)
    diff = (out_orig - out_traced).abs().max().item()
    print(f"  Verification: max diff = {diff:.2e}")
    assert diff < 1e-4, f"Output difference too large: {diff}"

    output_path = OUTPUT_DIR / f"swinir_real_x{scale}_traced.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> None:
    for scale in [2, 4]:
        export(scale)
    print("\nDone!")


if __name__ == "__main__":
    main()
