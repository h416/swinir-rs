# swinir-rs

AI video/image upscaling on Apple Silicon using [SwinIR](https://github.com/JingyunLiang/SwinIR) (Swin Transformer for Image Restoration). Powered by TorchScript and [tch-rs](https://github.com/LaurentMazare/tch-rs) with MPS GPU acceleration.

## Features

- **SwinIR super-resolution** -- x2 and x4 upscaling for real-world video and PNG images
- **MPS GPU acceleration** -- runs on Apple Silicon GPU via Metal Performance Shaders
- **BFloat16 inference** -- optional BF16 mode for faster processing on M3+ chips
- **Tiled processing** -- handles large frames with configurable tile size and overlap blending
- **ffmpeg fallback** -- bicubic, lanczos, and spline algorithms via ffmpeg (no GPU required)
- **Self-contained binary** -- libtorch dylibs are bundled automatically; no `DYLD_LIBRARY_PATH` or venv needed

## Requirements

- macOS Apple Silicon (M1 or later)
- Rust toolchain
- ffmpeg (`brew install ffmpeg`)
- Python 3 + PyTorch (only for model export)

## Setup

```bash
git clone https://github.com/h416/swinir-rs.git
cd swinir-rs
```

### 1. Download pre-traced models

Download the TorchScript models from [GitHub Releases](https://github.com/h416/swinir-rs/releases/tag/v0.1.0) and place them in `weights/`:

```bash
curl -L -o weights/swinir_real_x2_traced.pt https://github.com/h416/swinir-rs/releases/download/v0.1.0/swinir_real_x2_traced.pt
curl -L -o weights/swinir_real_x4_traced.pt https://github.com/h416/swinir-rs/releases/download/v0.1.0/swinir_real_x4_traced.pt
```

Alternatively, export models yourself (see [Model Export](#model-export) below).

### 2. Build

```bash
cargo build --release
```

On the first build, libtorch (~300MB) is downloaded automatically. The build script copies dylibs to `target/release/lib/` and sets rpath, so no environment variables are needed.

## Usage

```bash
./target/release/swinir-rs INPUT OUTPUT SCALE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT`  | Input file path (MP4 or PNG) |
| `OUTPUT` | Output file path (MP4 or PNG) |
| `SCALE`  | Scale factor: `x2`, `x3`, `x4` |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-a`, `--algorithm` | `swinir` | Algorithm: `swinir`, `bicubic`, `lanczos`, `spline` |
| `--crf` | `18` | Output quality (0 = lossless, 51 = lowest) |
| `-m`, `--model` | auto | Custom SwinIR model path (.pt) |
| `--tile-size` | `256` | Tile size for SwinIR (must be multiple of 8, 0 = no tiling) |
| `--bf16` | off | Use BFloat16 precision (M3+ chips recommended) |
| `--profile` | off | Print per-frame profiling breakdown |

### Tile Size

`--tile-size` controls how the frame is split for processing. Setting it to the shorter side of the input video reduces tiling overhead and improves speed.

For example, with a 640x360 input:
```bash
# Match the shorter side (360), rounded down to a multiple of 8
./target/release/swinir-rs input.mp4 output.mp4 x2 --tile-size 360
```

Use `--tile-size 0` to process the entire frame at once (requires more GPU memory).

### Examples

```bash
# SwinIR x2 upscale (default algorithm)
./target/release/swinir-rs input.mp4 output.mp4 x2

# SwinIR x4 upscale with high quality
./target/release/swinir-rs input.mp4 output.mp4 x4 --crf 12

# BFloat16 inference (faster on M3+)
./target/release/swinir-rs input.mp4 output.mp4 x2 --bf16

# lanczos interpolation via ffmpeg (no GPU needed)
./target/release/swinir-rs input.mp4 output.mp4 x2 -a lanczos

# PNG image upscale (SwinIR only)
./target/release/swinir-rs input.png output.png x2
./target/release/swinir-rs input.png output.png x4 --bf16
```

## Algorithms

| Algorithm | Method | GPU | Supported scales |
|-----------|--------|-----|------------------|
| `bicubic` | ffmpeg `-vf scale` | No | x2, x3, x4 |
| `lanczos` | ffmpeg `-vf scale` | No | x2, x3, x4 |
| `spline`  | ffmpeg `-vf scale` | No | x2, x3, x4 |
| `swinir`  | TorchScript inference | MPS (auto) | x2, x4 |

SwinIR automatically uses MPS GPU when available, falling back to CPU otherwise.

## Performance

SwinIR x2 upscale, 640x360 input, Apple M4 Pro:

| Method | Speed |
|--------|-------|
| MPS (GPU) | ~5.2 s/frame |

## Model Export

If you want to export the TorchScript models yourself instead of downloading them:

1. Install Python dependencies:
   ```bash
   pip install torch timm
   ```

2. Download SwinIR pretrained weights from the [official repository](https://github.com/JingyunLiang/SwinIR/releases):
   - `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth` -> `weights/swinir_real_x2_gan.pth`
   - `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` -> `weights/swinir_real_x4_gan.pth`

3. Run the export script:
   ```bash
   python scripts/export_torchscript.py
   ```

   This generates `weights/swinir_real_x2_traced.pt` and `weights/swinir_real_x4_traced.pt`.

## How It Works

### Video (MP4)
1. Extract video frames as PNG using ffmpeg (with VideoToolbox hardware decoding)
2. Upscale each frame through the SwinIR TorchScript model on MPS GPU
3. Extract audio from the original video
4. Recombine upscaled frames + audio into the output MP4

### Image (PNG)
1. Load the PNG image
2. Upscale through the SwinIR TorchScript model on MPS GPU
3. Save the upscaled image directly

For large frames/images, the tiled processing mode splits into overlapping tiles, processes them individually, and blends the results using linear weighting in the overlap regions.

## License

Apache-2.0

## Acknowledgments

- [SwinIR](https://github.com/JingyunLiang/SwinIR) by Liang et al. -- the original SwinIR model and pretrained weights
- [tch-rs](https://github.com/LaurentMazare/tch-rs) -- Rust bindings for PyTorch/libtorch
- [Claude Code](https://claude.ai/code) -- AI coding assistant used in development
