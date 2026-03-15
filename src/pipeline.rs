use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::Instant;

use crate::ffmpeg;
use crate::inference::SwinIRModel;

pub fn swinir_scale(input: &Path, output: &Path, factor: u32, crf: u32, model_override: Option<&Path>, tile_size: Option<u32>, profile: bool, bf16: bool) -> Result<()> {
    if factor != 2 && factor != 4 {
        anyhow::bail!("SwinIR does not support x{}. Only x2/x4 are supported.", factor);
    }

    let info = ffmpeg::get_video_info(input)?;
    let precision = if bf16 { "BF16" } else { "FP32" };
    println!(
        "SwinIR x{} crf={} {}: {} -> {}",
        factor, crf, precision,
        input.display(), output.display()
    );
    println!(
        "  FPS: {}, Frames: {}, Audio: {}",
        info.fps,
        info.nb_frames,
        if info.has_audio { "yes" } else { "no" }
    );

    // Determine TorchScript model path
    // Search for weights directory relative to the executable
    // (so the binary works when deployed to another machine)
    let default_path;
    let model_path = match model_override {
        Some(p) => p,
        None => {
            let exe_dir = std::env::current_exe()
                .context("Failed to get executable path")?
                .parent()
                .context("Executable parent directory not found")?
                .to_path_buf();
            // Search for weights/ relative to exe, fall back to CARGO_MANIFEST_DIR
            let candidates = [
                exe_dir.join("weights"),
                exe_dir.join("../weights"),
                exe_dir.join("../../weights"),
                exe_dir.join("../../../weights"),
            ];
            let weights_dir = candidates.iter()
                .map(|c| c.join(format!("swinir_real_x{}_traced.pt", factor)))
                .find(|p| p.exists());
            default_path = match weights_dir {
                Some(p) => p,
                None => {
                    // Final fallback: CARGO_MANIFEST_DIR (for development)
                    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
                        .join("weights")
                        .join(format!("swinir_real_x{}_traced.pt", factor))
                }
            };
            &default_path
        }
    };
    if !model_path.exists() {
        anyhow::bail!(
            "TorchScript model not found: {}\nPlease run export_torchscript.py first.",
            model_path.display()
        );
    }

    if let Some(ts) = tile_size {
        if ts != 0 && ts % 8 != 0 {
            anyhow::bail!("Tile size must be 0 (no tiling) or a multiple of 8, got {}", ts);
        }
    }

    println!("Loading model...");
    let mut model = SwinIRModel::new(&model_path, factor, tile_size, profile, bf16)?;

    let tmpdir = tempfile::tempdir().context("Failed to create temporary directory")?;
    println!("Temporary directory: {}", tmpdir.path().display());
    let frames_dir = tmpdir.path().join("frames");
    let upscaled_dir = tmpdir.path().join("upscaled");
    std::fs::create_dir(&frames_dir)?;
    std::fs::create_dir(&upscaled_dir)?;

    // Extract frames
    println!("Extracting frames...");
    ffmpeg::extract_frames(input, &frames_dir, &info.fps)?;

    let mut frame_files: Vec<_> = std::fs::read_dir(&frames_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    frame_files.sort();

    let total = frame_files.len();
    println!("  Extracted {} frames", total);

    // Upscale frames one by one
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Upscaling: [{bar:40}] {pos}/{len} ({eta}) {msg}")?
    );

    let upscale_start = Instant::now();
    for frame_path in &frame_files {
        let frame_start = Instant::now();

        let img = image::open(frame_path)
            .with_context(|| format!("Failed to load frame: {}", frame_path.display()))?
            .to_rgb8();

        let upscaled = model.upscale(&img)?;

        let out_name = frame_path.file_name().unwrap();
        upscaled.save(upscaled_dir.join(out_name))?;

        // Delete original frame to save disk space
        std::fs::remove_file(frame_path)?;

        let elapsed = frame_start.elapsed();
        pb.set_message(format!("{:.2}s/frame", elapsed.as_secs_f64()));
        pb.inc(1);
    }
    pb.finish_and_clear();
    let total_elapsed = upscale_start.elapsed();
    println!(
        "Upscaling: {}/{} done ({:.1}s, avg {:.2}s/frame)",
        total, total, total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / total as f64,
    );

    // Extract audio
    let audio_path = if info.has_audio {
        let p = tmpdir.path().join("audio.aac");
        ffmpeg::extract_audio(input, &p)?;
        Some(p)
    } else {
        None
    };

    // Combine frames and audio
    println!("Combining video...");
    ffmpeg::combine(
        &upscaled_dir,
        audio_path.as_deref(),
        output,
        &info.fps,
        crf,
    )?;

    println!("Done: {}", output.display());
    Ok(())
}
