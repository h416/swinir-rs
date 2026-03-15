mod ffmpeg;
mod inference;
mod pipeline;

use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};

const FFMPEG_ALGORITHMS: &[&str] = &["bicubic", "lanczos", "spline"];

#[derive(Parser)]
#[command(about = "Upscale video")]
struct Cli {
    /// Input MP4 path
    input: PathBuf,
    /// Output MP4 path
    output: PathBuf,
    /// Scale factor (x2, x3, x4)
    scale: String,
    /// Upscale algorithm (bicubic, lanczos, spline, swinir)
    #[arg(short = 'a', long = "algorithm", default_value = "swinir")]
    algorithm: String,
    /// Output quality (0=lossless, 18=visually near-lossless)
    #[arg(long, default_value_t = 18)]
    crf: u32,
    /// SwinIR model path (.pt). Defaults to weights/swinir_real_x{scale}_traced.pt
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,
    /// Tile size for SwinIR (must be multiple of 8, default: 256)
    #[arg(long = "tile-size", default_value_t = 256)]
    tile_size: u32,
    /// Print per-frame profiling breakdown (first frame only)
    #[arg(long)]
    profile: bool,
    /// Use BFloat16 precision for reduced memory usage (requires M3+ chip)
    #[arg(long)]
    bf16: bool,
}

fn parse_scale(s: &str) -> Result<u32> {
    match s {
        "x2" => Ok(2),
        "x3" => Ok(3),
        "x4" => Ok(4),
        _ => anyhow::bail!("Unsupported scale factor: {} (only x2, x3, x4)", s),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let factor = parse_scale(&cli.scale)?;

    let is_png = |p: &Path| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"));
    if is_png(&cli.input) && is_png(&cli.output) {
        if cli.algorithm != "swinir" {
            anyhow::bail!("PNG image upscaling only supports swinir algorithm");
        }
        pipeline::swinir_image_scale(&cli.input, &cli.output, factor, cli.model.as_deref(), Some(cli.tile_size), cli.profile, cli.bf16)?;
    } else if FFMPEG_ALGORITHMS.contains(&cli.algorithm.as_str()) {
        ffmpeg::ffmpeg_scale(&cli.input, &cli.output, factor, &cli.algorithm, cli.crf)?;
    } else if cli.algorithm == "swinir" {
        pipeline::swinir_scale(&cli.input, &cli.output, factor, cli.crf, cli.model.as_deref(), Some(cli.tile_size), cli.profile, cli.bf16)?;
    } else {
        anyhow::bail!(
            "Unsupported algorithm: {} (bicubic, lanczos, spline, swinir)",
            cli.algorithm
        );
    }

    Ok(())
}
