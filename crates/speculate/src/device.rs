//! Device selection helpers.

use crate::Result;
use candle_core::Device;

/// Pick the best available device, preferring (in order) CUDA, Metal, then CPU.
///
/// Honors the `ABYO_SPECULATE_DEVICE` env var if set: `cpu`, `cuda`, `metal`,
/// or `cuda:N` to pin a specific GPU index.
pub fn auto_device() -> Result<Device> {
    if let Ok(spec) = std::env::var("ABYO_SPECULATE_DEVICE") {
        return parse_device_spec(&spec);
    }

    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return Ok(d);
    }

    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return Ok(d);
    }

    Ok(Device::Cpu)
}

fn parse_device_spec(spec: &str) -> Result<Device> {
    let spec = spec.trim().to_lowercase();
    match spec.as_str() {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::new_cuda(0)?),
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::new_metal(0)?),
        s if s.starts_with("cuda:") => {
            #[cfg(feature = "cuda")]
            {
                let idx: usize = s.trim_start_matches("cuda:").parse().map_err(
                    |e: std::num::ParseIntError| {
                        crate::Error::Other(anyhow::anyhow!("bad cuda index: {e}"))
                    },
                )?;
                Ok(Device::new_cuda(idx)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = s;
                Err(crate::Error::Other(anyhow::anyhow!(
                    "CUDA requested but crate built without `cuda` feature"
                )))
            }
        }
        other => Err(crate::Error::Other(anyhow::anyhow!(
            "unknown device spec: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The two env-var paths are tested via `parse_device_spec` directly so
    // that parallel test execution does not race on the shared
    // `ABYO_SPECULATE_DEVICE` global.

    #[test]
    fn parse_explicit_cpu_spec() {
        let d = parse_device_spec("cpu").unwrap();
        assert!(matches!(d, Device::Cpu));
    }

    #[test]
    fn parse_uppercase_cpu_spec() {
        let d = parse_device_spec("CPU").unwrap();
        assert!(matches!(d, Device::Cpu));
    }

    #[test]
    fn parse_unknown_spec_errors() {
        assert!(parse_device_spec("not-a-device").is_err());
    }
}
