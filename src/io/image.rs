use crate::error::ImageOutputError;
use image::ImageBuffer;
use std::path::Path;

/// Saves a u32 (0RGB) buffer to a PNG file.
pub fn save_buffer_to_image<P: AsRef<Path>>(
    buffer: &[u32],
    width: usize,
    height: usize,
    path: P,
) -> Result<(), ImageOutputError> {
    let image_width =
        u32::try_from(width).map_err(|_| ImageOutputError::InvalidDimensions { width, height })?;
    let image_height =
        u32::try_from(height).map_err(|_| ImageOutputError::InvalidDimensions { width, height })?;
    let expected = width
        .checked_mul(height)
        .ok_or(ImageOutputError::InvalidDimensions { width, height })?;
    if buffer.len() != expected {
        return Err(ImageOutputError::BufferLength {
            expected,
            actual: buffer.len(),
        });
    }

    let output_path = path.as_ref();
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|source| ImageOutputError::CreateParent {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let mut img_buf = ImageBuffer::new(image_width, image_height);

    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let idx = (y as usize) * width + (x as usize);
        let color_u32 = buffer[idx];

        let r = ((color_u32 >> 16) & 0xFF) as u8;
        let g = ((color_u32 >> 8) & 0xFF) as u8;
        let b = (color_u32 & 0xFF) as u8;

        *pixel = image::Rgb([r, g, b]);
    }

    img_buf
        .save(output_path)
        .map_err(|source| ImageOutputError::Save {
            path: output_path.to_path_buf(),
            source,
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_buffer_length() {
        let error = save_buffer_to_image(&[], 1, 1, "unused.png").unwrap_err();
        assert!(matches!(error, ImageOutputError::BufferLength { .. }));
    }
}
