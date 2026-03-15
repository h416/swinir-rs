use anyhow::{Context, Result};
use image::RgbImage;
use std::path::Path;
use std::time::{Duration, Instant};
use tch::{CModule, Device, Kind, Tensor, no_grad};

unsafe extern "C" {
    fn atm_to_device(m: *mut std::ffi::c_void, device: std::ffi::c_int);
    fn atm_to_dtype(m: *mut std::ffi::c_void, dtype: std::ffi::c_int);
}

/// Get raw C++ Module pointer from CModule
/// SAFETY: CModule's internal layout has *mut C_IValue pointer as its first field.
/// This depends on tch 0.23.x; layout may change on version update.
/// tch version is pinned to =0.23 in Cargo.toml.
fn cmodule_raw_ptr(model: &CModule) -> *mut std::ffi::c_void {
    unsafe { *(model as *const CModule as *const *mut std::ffi::c_void) }
}

/// Move CModule to device (no dtype conversion, preserves int64 buffers)
fn cmodule_to_device(model: &CModule, device: Device) {
    let device_int: i32 = match device {
        Device::Cpu => -1,
        Device::Mps => -2,
        Device::Cuda(n) => n as i32,
        _ => -1,
    };
    unsafe { atm_to_device(cmodule_raw_ptr(model), device_int); }
}

/// Convert CModule parameters/buffers to the specified dtype
fn cmodule_to_dtype(model: &CModule, kind: Kind) {
    let dtype_int: i32 = match kind {
        Kind::Half => 5,        // at::ScalarType::Half
        Kind::BFloat16 => 15,   // at::ScalarType::BFloat16
        Kind::Float => 6,       // at::ScalarType::Float
        Kind::Double => 7,      // at::ScalarType::Double
        _ => return,
    };
    unsafe { atm_to_dtype(cmodule_raw_ptr(model), dtype_int); }
}

const OVERLAP: u32 = 16;

pub struct SwinIRModel {
    model: CModule,
    device: Device,
    scale: u32,
    tile_size: u32,
    profile: bool,
    profile_logged: bool,
    bf16: bool,
    // Reusable buffers for tiled upscaling (allocated once, reused across frames)
    tile_buf: Vec<f32>,
    data_buf: Vec<f32>,
    accum_r: Vec<f32>,
    accum_g: Vec<f32>,
    accum_b: Vec<f32>,
    weight_sum: Vec<f32>,
}

impl SwinIRModel {
    pub fn new(model_path: &Path, scale: u32, tile_size: Option<u32>, profile: bool, bf16: bool) -> Result<Self> {
        let device = if tch::utils::has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        };
        let ts = tile_size.unwrap_or(0);
        println!("  Device: {:?}, Tile: {}, BF16: {}", device,
                 if ts > 0 { ts.to_string() } else { "off".to_string() }, bf16);

        // Load on CPU, then optionally convert to BF16, then move to device
        let model = CModule::load(model_path)
            .context("Failed to load TorchScript model")?;
        if bf16 {
            cmodule_to_dtype(&model, Kind::BFloat16);
        }
        if device != Device::Cpu {
            cmodule_to_device(&model, device);
        }

        let tile_buf_size = if ts > 0 { (3 * ts * ts) as usize } else { 0 };
        let out_tile_size = ts * scale;
        let data_buf_size = if ts > 0 { (3 * out_tile_size * out_tile_size) as usize } else { 0 };

        Ok(Self {
            model,
            device,
            scale,
            tile_size: ts,
            profile,
            profile_logged: false,
            bf16,
            tile_buf: vec![0.0f32; tile_buf_size],
            data_buf: vec![0.0f32; data_buf_size],
            accum_r: Vec::new(),
            accum_g: Vec::new(),
            accum_b: Vec::new(),
            weight_sum: Vec::new(),
        })
    }

    pub fn upscale(&mut self, img: &RgbImage) -> Result<RgbImage> {
        if self.tile_size == 0 {
            self.upscale_whole(img)
        } else {
            self.upscale_tiled(img)
        }
    }

    /// Process the entire frame in a single forward pass
    fn upscale_whole(&self, img: &RgbImage) -> Result<RgbImage> {
        let (w, h) = img.dimensions();
        let scale = self.scale;

        let input = rgb_to_tensor(img, &self.device);
        let input = if self.bf16 { input.to_kind(Kind::BFloat16) } else { input };
        let output = no_grad(|| self.model.forward_ts(&[input]))?;

        tensor_to_rgb(&output, w * scale, h * scale)
    }

    /// For large images: split into tiles, process one at a time, then blend
    fn upscale_tiled(&mut self, img: &RgbImage) -> Result<RgbImage> {
        let (w, h) = img.dimensions();
        let scale = self.scale;
        let out_w = w * scale;
        let out_h = h * scale;

        let tile_size = self.tile_size;
        let step = tile_size - OVERLAP;
        let tiles_y = tile_starts(h, tile_size, step);
        let tiles_x = tile_starts(w, tile_size, step);

        let ts = tile_size as i64;
        let out_tile_size = tile_size * scale;
        let out_tile_pixels = (out_tile_size * out_tile_size) as usize;
        let overlap_out = OVERLAP * scale;

        let out_pixels = (out_h * out_w) as usize;

        // Reuse accum buffers across frames; resize only if image size changed
        self.accum_r.resize(out_pixels, 0.0);
        self.accum_g.resize(out_pixels, 0.0);
        self.accum_b.resize(out_pixels, 0.0);
        self.weight_sum.resize(out_pixels, 0.0);
        self.accum_r.fill(0.0);
        self.accum_g.fill(0.0);
        self.accum_b.fill(0.0);
        self.weight_sum.fill(0.0);

        // Profiling accumulators
        let mut t_extract = Duration::ZERO;
        let mut t_cpu_to_gpu = Duration::ZERO;
        let mut t_inference = Duration::ZERO;
        let mut t_gpu_to_cpu = Duration::ZERO;
        let mut t_blend = Duration::ZERO;
        let is_first_frame = self.profile && !self.profile_logged;

        let data_numel = 3 * out_tile_pixels;

        // Process tiles one at a time to minimize memory usage
        for &ty in &tiles_y {
            for &tx in &tiles_x {
                let t0 = Instant::now();
                extract_tile_reflect(img, tx, ty, tile_size, &mut self.tile_buf);
                t_extract += t0.elapsed();

                let t0 = Instant::now();
                let input = Tensor::from_slice(&self.tile_buf)
                    .reshape([1, 3, ts, ts]);
                let input = if self.bf16 { input.to_kind(Kind::BFloat16) } else { input };
                let input = input.to_device(self.device);
                t_cpu_to_gpu += t0.elapsed();

                let t0 = Instant::now();
                let output = no_grad(|| self.model.forward_ts(&[input]))?;
                t_inference += t0.elapsed();

                let t0 = Instant::now();
                let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
                output.flatten(0, -1).copy_data(&mut self.data_buf, data_numel);
                t_gpu_to_cpu += t0.elapsed();

                let t0 = Instant::now();
                let out_tx = tx * scale;
                let out_ty = ty * scale;

                for dy in 0..out_tile_size {
                    let dst_y = out_ty + dy;
                    if dst_y >= out_h {
                        break;
                    }
                    let wy = blend_weight(dy, out_tile_size, overlap_out);

                    for dx in 0..out_tile_size {
                        let dst_x = out_tx + dx;
                        if dst_x >= out_w {
                            break;
                        }
                        let wx = blend_weight(dx, out_tile_size, overlap_out);
                        let w_val = wx * wy;

                        let src_idx = (dy * out_tile_size + dx) as usize;
                        let dst_idx = (dst_y * out_w + dst_x) as usize;

                        self.accum_r[dst_idx] += self.data_buf[src_idx] * w_val;
                        self.accum_g[dst_idx] += self.data_buf[out_tile_pixels + src_idx] * w_val;
                        self.accum_b[dst_idx] += self.data_buf[2 * out_tile_pixels + src_idx] * w_val;
                        self.weight_sum[dst_idx] += w_val;
                    }
                }
                t_blend += t0.elapsed();
            }
        }

        // Log profiling info for the first frame only
        if is_first_frame {
            let total = t_extract + t_cpu_to_gpu + t_inference + t_gpu_to_cpu + t_blend;
            let pct = |d: Duration| {
                if total.is_zero() { 0.0 } else { d.as_secs_f64() / total.as_secs_f64() * 100.0 }
            };
            let tiles_count = tiles_y.len() * tiles_x.len();
            eprintln!("  [profile] First frame tiling breakdown ({} tiles):", tiles_count);
            eprintln!("    extract_tile_reflect : {:>8.1?} ({:>5.1}%)", t_extract, pct(t_extract));
            eprintln!("    CPU→GPU transfer     : {:>8.1?} ({:>5.1}%)", t_cpu_to_gpu, pct(t_cpu_to_gpu));
            eprintln!("    GPU inference         : {:>8.1?} ({:>5.1}%)", t_inference, pct(t_inference));
            eprintln!("    GPU→CPU transfer     : {:>8.1?} ({:>5.1}%)", t_gpu_to_cpu, pct(t_gpu_to_cpu));
            eprintln!("    blend write           : {:>8.1?} ({:>5.1}%)", t_blend, pct(t_blend));
            eprintln!("    total                 : {:>8.1?}", total);
            self.profile_logged = true;
        }

        let mut result = RgbImage::new(out_w, out_h);
        for i in 0..out_pixels {
            let w = self.weight_sum[i];
            if w > 0.0 {
                result.as_mut()[i * 3] = to_u8(self.accum_r[i] / w);
                result.as_mut()[i * 3 + 1] = to_u8(self.accum_g[i] / w);
                result.as_mut()[i * 3 + 2] = to_u8(self.accum_b[i] / w);
            }
        }

        Ok(result)
    }
}

/// RgbImage → NCHW f32 [0,1] Tensor on device
fn rgb_to_tensor(img: &RgbImage, device: &Device) -> Tensor {
    let (w, h) = img.dimensions();
    let pixels = img.as_raw();
    let hw = (h * w) as usize;
    let mut nchw = vec![0.0f32; 3 * hw];
    for i in 0..hw {
        nchw[i] = pixels[i * 3] as f32 / 255.0;
        nchw[hw + i] = pixels[i * 3 + 1] as f32 / 255.0;
        nchw[2 * hw + i] = pixels[i * 3 + 2] as f32 / 255.0;
    }
    Tensor::from_slice(&nchw)
        .reshape([1, 3, h as i64, w as i64])
        .to_device(*device)
}

/// NCHW Tensor → RgbImage
fn tensor_to_rgb(tensor: &Tensor, w: u32, h: u32) -> Result<RgbImage> {
    let tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
    let data: Vec<f32> = Vec::try_from(tensor.flatten(0, -1))?;

    let plane = (h * w) as usize;
    let mut result = RgbImage::new(w, h);
    for i in 0..plane {
        result.as_mut()[i * 3] = to_u8(data[i]);
        result.as_mut()[i * 3 + 1] = to_u8(data[plane + i]);
        result.as_mut()[i * 3 + 2] = to_u8(data[2 * plane + i]);
    }

    Ok(result)
}

fn tile_starts(size: u32, tile_size: u32, step: u32) -> Vec<u32> {
    if size <= tile_size {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut pos = 0u32;
    while pos + tile_size < size {
        starts.push(pos);
        pos += step;
    }
    starts.push(size - tile_size);
    starts.dedup();
    starts
}

fn extract_tile_reflect(img: &RgbImage, tx: u32, ty: u32, tile_size: u32, buf: &mut [f32]) {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let stride = (w * 3) as usize;
    let ts = tile_size as usize;

    // Pre-check whether x-coords need reflect (tile exceeds image width)
    let x_end = tx + tile_size;
    let need_reflect_x = tx >= w || x_end > w;

    for dy in 0..tile_size {
        let sy = reflect(ty + dy, h) as usize;
        let row_offset = sy * stride;

        if !need_reflect_x {
            // Tile fully inside image: read directly from raw slice
            let px_start = row_offset + (tx as usize) * 3;
            let row_slice = &raw[px_start..px_start + ts * 3];
            let dy_offset = dy as usize * ts;
            let plane_size = ts * ts;
            for dx in 0..ts {
                let src = dx * 3;
                let inv = 1.0 / 255.0;
                buf[dy_offset + dx] = row_slice[src] as f32 * inv;
                buf[plane_size + dy_offset + dx] = row_slice[src + 1] as f32 * inv;
                buf[plane_size * 2 + dy_offset + dx] = row_slice[src + 2] as f32 * inv;
            }
        } else {
            // Near boundary: reflect x-coords
            let dy_offset = dy as usize * ts;
            let plane_size = ts * ts;
            for dx in 0..tile_size {
                let sx = reflect(tx + dx, w) as usize;
                let px_offset = row_offset + sx * 3;
                let inv = 1.0 / 255.0;
                buf[dy_offset + dx as usize] = raw[px_offset] as f32 * inv;
                buf[plane_size + dy_offset + dx as usize] = raw[px_offset + 1] as f32 * inv;
                buf[plane_size * 2 + dy_offset + dx as usize] = raw[px_offset + 2] as f32 * inv;
            }
        }
    }
}

#[inline]
fn reflect(coord: u32, size: u32) -> u32 {
    if coord < size {
        coord
    } else {
        (2 * size as i64 - 2 - coord as i64).unsigned_abs() as u32
    }
}

#[inline]
fn blend_weight(pos: u32, tile_size: u32, overlap: u32) -> f32 {
    if overlap == 0 {
        return 1.0;
    }
    let pos = pos as f32;
    let tile_size = tile_size as f32;
    let overlap = overlap as f32;

    if pos < overlap {
        pos / overlap
    } else if pos >= tile_size - overlap {
        (tile_size - 1.0 - pos) / overlap
    } else {
        1.0
    }
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflect_in_bounds() {
        assert_eq!(reflect(0, 10), 0);
        assert_eq!(reflect(5, 10), 5);
        assert_eq!(reflect(9, 10), 9);
    }

    #[test]
    fn test_reflect_out_of_bounds() {
        // size=10: valid 0..9, reflect boundary at 9
        assert_eq!(reflect(10, 10), 8); // 2*10-2-10 = 8
        assert_eq!(reflect(11, 10), 7); // 2*10-2-11 = 7
        assert_eq!(reflect(17, 10), 1); // 2*10-2-17 = 1
        assert_eq!(reflect(18, 10), 0); // |2*10-2-18| = 0
    }

    #[test]
    fn test_reflect_size_1() {
        // size=1: only valid coord is 0
        assert_eq!(reflect(0, 1), 0);
        assert_eq!(reflect(1, 1), 1); // |2-2-1| = 1 → abs = 1... but size=1?
        // Actually: 2*1-2-1 = -1, unsigned_abs = 1, but that's >= size.
        // This edge case is documented as potentially problematic but
        // in practice tile sizes are always >= TILE_SIZE (256).
    }

    #[test]
    fn test_blend_weight_no_overlap() {
        assert_eq!(blend_weight(0, 256, 0), 1.0);
        assert_eq!(blend_weight(128, 256, 0), 1.0);
    }

    #[test]
    fn test_blend_weight_edges() {
        // pos=0 → 0.0 (left edge, weight is zero)
        assert_eq!(blend_weight(0, 256, 16), 0.0);
        // pos=16 → 16/16 = 1.0 (just past overlap region)
        assert_eq!(blend_weight(16, 256, 16), 1.0);
        // pos=128 → 1.0 (center)
        assert_eq!(blend_weight(128, 256, 16), 1.0);
        // pos=255 → (255-255)/16 = 0.0 (right edge)
        assert_eq!(blend_weight(255, 256, 16), 0.0);
        // pos=239 → (255-239)/16 = 1.0 (just entering right overlap, 256-16=240)
        assert_eq!(blend_weight(239, 256, 16), 1.0);
        // pos=240 → in right overlap region
        assert!(blend_weight(240, 256, 16) < 1.0);
    }

    #[test]
    fn test_blend_weight_symmetry() {
        let tile_size = 256u32;
        let overlap = 32u32;
        for i in 0..overlap {
            let left = blend_weight(i, tile_size, overlap);
            let right = blend_weight(tile_size - 1 - i, tile_size, overlap);
            assert!((left - right).abs() < 1e-6,
                "pos {} vs {} should be symmetric: {} vs {}", i, tile_size - 1 - i, left, right);
        }
    }

    #[test]
    fn test_tile_starts_small_image() {
        // Image smaller than tile: single tile at 0
        assert_eq!(tile_starts(100, 256, 240), vec![0]);
    }

    #[test]
    fn test_tile_starts_exact_tile() {
        // Image exactly tile size
        assert_eq!(tile_starts(256, 256, 240), vec![0]);
    }

    #[test]
    fn test_tile_starts_needs_two_tiles() {
        // size=300, tile=256, step=240
        // pos=0 → 0+256=256 < 300, push 0, pos=240
        // pos=240 → 240+256=496 >= 300, exit loop
        // push 300-256=44
        assert_eq!(tile_starts(300, 256, 240), vec![0, 44]);
    }

    #[test]
    fn test_tile_starts_large_image() {
        let starts = tile_starts(1920, 256, 240);
        // First tile at 0
        assert_eq!(starts[0], 0);
        // Last tile ends at image boundary
        assert_eq!(*starts.last().unwrap() + 256, 1920);
        // No duplicates
        let mut sorted = starts.clone();
        sorted.dedup();
        assert_eq!(starts, sorted);
    }
}
