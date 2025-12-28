use image::ImageBuffer;
use log::error;
use std::path::Path;

/// Saves a u32 (0RGB) buffer to a PNG file.
pub fn save_buffer_to_image(buffer: &[u32], width: usize, height: usize, path: &str) {
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let idx = (y as usize) * width + (x as usize);
        let color_u32 = buffer[idx];

        let r = ((color_u32 >> 16) & 0xFF) as u8;
        let g = ((color_u32 >> 8) & 0xFF) as u8;
        let b = (color_u32 & 0xFF) as u8;

        *pixel = image::Rgb([r, g, b]);
    }

    if let Err(e) = img_buf.save(Path::new(path)) {
        error!("Failed to save image to '{}': {}", path, e);
    }
}
