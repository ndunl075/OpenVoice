//! `cpal`-backed microphone capture feeding the always-on [`ring_buffer`].
//!
//! See `dictation-architecture.md` §2.1. Opens the default input device
//! exactly once, at daemon start, and keeps the stream alive for the life
//! of the process — this is what kills the 100-300ms mic-device-init
//! latency the naive pipeline pays on every hotkey press.
//!
//! Whatever native sample rate / channel count / sample format the device
//! reports, every callback is converted to mono `f32` at
//! [`ring_buffer::DEFAULT_SAMPLE_RATE_HZ`] before it's pushed into the ring
//! buffer, so nothing downstream needs to know what hardware is attached.

mod convert;
mod resample;

pub use convert::downmix_to_mono;
pub use resample::linear_resample;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use ring_buffer::SharedRingBuffer;

pub const TARGET_SAMPLE_RATE_HZ: u32 = ring_buffer::DEFAULT_SAMPLE_RATE_HZ;

#[derive(Debug, thiserror::Error)]
pub enum AudioCaptureError {
    #[error("no default input device found")]
    NoInputDevice,
    #[error("cpal error: {0}")]
    Cpal(#[from] cpal::Error),
    #[error("unsupported input sample format: {0:?}")]
    UnsupportedSampleFormat(SampleFormat),
}

/// Owns the live microphone stream. Dropping this stops capture — the
/// daemon holds one for its entire lifetime.
pub struct AudioCapture {
    stream: cpal::Stream,
    device_name: String,
    device_sample_rate_hz: u32,
    device_channels: u16,
}

impl AudioCapture {
    /// Opens the default input device and starts streaming into `ring`,
    /// converting every callback to mono audio at
    /// [`TARGET_SAMPLE_RATE_HZ`] first.
    pub fn start(ring: SharedRingBuffer) -> Result<Self, AudioCaptureError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioCaptureError::NoInputDevice)?;
        let device_name = device.to_string();

        let supported_config = device.default_input_config()?;
        let sample_format = supported_config.sample_format();
        let config: cpal::StreamConfig = supported_config.into();
        let device_sample_rate_hz = config.sample_rate;
        let device_channels = config.channels;

        let error_ring = ring.clone();
        let error_callback = move |err: cpal::Error| {
            // Nothing sane to do with a mid-stream device error beyond
            // surfacing it; the ring buffer just stops receiving new audio
            // until the daemon notices and restarts capture.
            eprintln!("audio input stream error: {err}");
            let _ = &error_ring; // keep the clone alive for future use (logging hook)
        };

        let stream = match sample_format {
            SampleFormat::F32 => build_stream::<f32>(
                &device,
                &config,
                ring,
                device_sample_rate_hz,
                device_channels,
                error_callback,
            )?,
            SampleFormat::I16 => build_stream::<i16>(
                &device,
                &config,
                ring,
                device_sample_rate_hz,
                device_channels,
                error_callback,
            )?,
            SampleFormat::U16 => build_stream::<u16>(
                &device,
                &config,
                ring,
                device_sample_rate_hz,
                device_channels,
                error_callback,
            )?,
            SampleFormat::I32 => build_stream::<i32>(
                &device,
                &config,
                ring,
                device_sample_rate_hz,
                device_channels,
                error_callback,
            )?,
            other => return Err(AudioCaptureError::UnsupportedSampleFormat(other)),
        };

        stream.play()?;

        Ok(Self {
            stream,
            device_name,
            device_sample_rate_hz,
            device_channels,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_sample_rate_hz(&self) -> u32 {
        self.device_sample_rate_hz
    }

    pub fn device_channels(&self) -> u16 {
        self.device_channels
    }

    /// Stops the stream. Equivalent to dropping the `AudioCapture`, spelled
    /// out for callers that want the intent explicit (e.g. the tray "pause
    /// listening" action).
    pub fn stop(self) {
        drop(self.stream);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    ring: SharedRingBuffer,
    device_sample_rate_hz: u32,
    device_channels: u16,
    error_callback: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<cpal::Stream, AudioCaptureError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let data_callback = move |data: &[T], _: &cpal::InputCallbackInfo| {
        let as_f32: Vec<f32> = data.iter().map(|&s| f32::from_sample(s)).collect();
        let mono = downmix_to_mono(&as_f32, device_channels);
        let resampled = linear_resample(&mono, device_sample_rate_hz, TARGET_SAMPLE_RATE_HZ);
        if let Ok(mut guard) = ring.lock() {
            guard.push(&resampled);
        }
    };

    Ok(device.build_input_stream(*config, data_callback, error_callback, None)?)
}
