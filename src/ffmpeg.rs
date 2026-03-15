use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";
const FFPROBE: &str = "/opt/homebrew/bin/ffprobe";

#[derive(Debug)]
pub struct VideoInfo {
    pub fps: String,
    pub nb_frames: u64,
    pub has_audio: bool,
}

#[derive(Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: String,
    avg_frame_rate: Option<String>,
    nb_frames: Option<String>,
}

pub fn get_video_info(input: &Path) -> Result<VideoInfo> {
    let output = Command::new(FFPROBE)
        .args(["-v", "quiet", "-print_format", "json", "-show_streams"])
        .arg(input)
        .output()
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let data: ProbeOutput = serde_json::from_slice(&output.stdout)
        .context("Failed to parse ffprobe output")?;

    let mut info = VideoInfo {
        fps: "30/1".to_string(),
        nb_frames: 0,
        has_audio: false,
    };

    for stream in &data.streams {
        match stream.codec_type.as_str() {
            "video" => {
                if let Some(ref fps) = stream.avg_frame_rate {
                    info.fps = fps.clone();
                }
                if let Some(ref nf) = stream.nb_frames {
                    info.nb_frames = nf.parse().unwrap_or(0);
                }
            }
            "audio" => {
                info.has_audio = true;
            }
            _ => {}
        }
    }

    Ok(info)
}

pub fn ffmpeg_scale(input: &Path, output: &Path, factor: u32, algorithm: &str, crf: u32) -> Result<()> {
    let info = get_video_info(input)?;
    let vf = format!("scale=iw*{}:ih*{}:flags={}", factor, factor, algorithm);

    println!("ffmpeg {} x{} crf={}: {} -> {}",
        algorithm, factor, crf,
        input.display(), output.display());

    let result = Command::new(FFMPEG)
        .args(["-y", "-i"])
        .arg(input)
        .args(["-vf", &vf, "-r", &info.fps])
        .args(["-c:v", "libx264", "-preset", "medium", "-crf", &crf.to_string()])
        .args(["-c:a", "copy"])
        .arg(output)
        .output()
        .context("Failed to run ffmpeg")?;

    if !result.status.success() {
        bail!("ffmpeg failed: {}", String::from_utf8_lossy(&result.stderr));
    }
    Ok(())
}

pub fn extract_frames(input: &Path, frames_dir: &Path, fps: &str) -> Result<()> {
    let pattern = frames_dir.join("%08d.png");
    let mut cmd = Command::new(FFMPEG);
    cmd.arg("-y");
    #[cfg(target_os = "macos")]
    cmd.args(["-hwaccel", "videotoolbox"]);
    cmd.arg("-i").arg(input).args(["-r", fps]).arg(&pattern);
    let output = cmd
        .output()
        .context("Failed to extract frames")?;

    if !output.status.success() {
        bail!("Frame extraction error: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

pub fn extract_audio(input: &Path, audio_path: &Path) -> Result<()> {
    let output = Command::new(FFMPEG)
        .args(["-y", "-i"])
        .arg(input)
        .args(["-vn", "-acodec", "copy"])
        .arg(audio_path)
        .output()
        .context("Failed to extract audio")?;

    if !output.status.success() {
        bail!("Audio extraction error: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

pub fn combine(
    frames_dir: &Path,
    audio_path: Option<&Path>,
    output: &Path,
    fps: &str,
    crf: u32,
) -> Result<()> {
    let pattern = frames_dir.join("%08d.png");
    let mut cmd = Command::new(FFMPEG);
    cmd.args(["-y", "-framerate", fps, "-i"])
        .arg(&pattern);

    if let Some(audio) = audio_path {
        cmd.args(["-i".as_ref(), audio.as_os_str()]);
    }

    cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", &crf.to_string()])
        .args(["-pix_fmt", "yuv420p"]);

    if audio_path.is_some() {
        cmd.args(["-c:a", "copy", "-shortest"]);
    }

    cmd.arg(output);

    let output = cmd.output().context("Failed to combine video")?;
    if !output.status.success() {
        bail!("Video combining error: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}
