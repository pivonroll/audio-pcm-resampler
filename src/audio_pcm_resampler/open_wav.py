from __future__ import annotations

import dataclasses
from os import PathLike
from typing import Union

from numpy.typing import NDArray


@dataclasses.dataclass
class WavParams:
    nchannels: int
    """Number of channels."""
    sampwidth: int
    """Sample width."""
    framerate: int
    """Sample rate."""
    nframes: int
    """Number of frames."""
    comptype: str
    """Compression type."""
    compname: str
    """Compression name."""

    @classmethod
    def from_wave_params(cls, wave_params) -> WavParams:
        return cls(
            nchannels=wave_params.nchannels,
            sampwidth=wave_params.sampwidth,
            framerate=wave_params.framerate,
            nframes=wave_params.nframes,
            comptype=wave_params.comptype,
            compname=wave_params.compname,
        )

@dataclasses.dataclass
class WavResult:
    params: WavParams
    frames: bytes

def open_wav_file(file_path: str) -> WavResult:
    import wave

    with wave.open(file_path, 'rb') as wav_file:
        wav_params = wav_file.getparams()
        wav_frames = wav_file.readframes(wav_params.nframes)
        return WavResult(params=WavParams.from_wave_params(wav_params), frames=wav_frames)

def chunk_bytes(data, chunk_size=4096):
    """Generator that yields chunks of bytes from the given data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def save_int_wav_file(file_path: str, bytes_data: bytes, sample_rate: int, bit_depth: int, channels: int):
    import wave

    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes_data)


import struct
import numpy as np  # For generating sample data (optional)


def write_float_wav(file_path: PathLike,
                    samples: Union[NDArray[np.float32], NDArray[np.float64]],
                    sample_rate: int=44100,
                    num_channels: int=1):
    """Write 32-bit float PCM WAV file.

    Args:
        file_path (str): Output file path.
        samples (np.ndarray or list): Audio samples (mono: 1D array; stereo: 2D array [samples, channels]).
        sample_rate (int): Sample rate (Hz).
        num_channels (int): Number of channels (1 or 2).
    """
    # Ensure samples are in interleaved format (for stereo) and 32-bit float
    if num_channels == 2 and samples.ndim == 2:
        samples = samples.flatten(order='F')  # Interleave: [L0, R0, L1, R1, ...]
    samples = np.asarray(samples, dtype=np.float32)  # Enforce 32-bit float

    # Calculate chunk sizes
    subchunk2_size = len(samples) * 4  # 4 bytes per sample
    riff_size = 36 + subchunk2_size  # RIFF chunk size (total file size - 8)

    # Pack data into binary format using struct
    with open(file_path, 'wb') as f:
        # RIFF Chunk
        f.write(struct.pack('<4sI4s', b'RIFF', riff_size, b'WAVE'))

        # fmt Subchunk
        f.write(struct.pack('<4sIHHIIHH',
                            b'fmt ',  # Subchunk1ID
                            16,  # Subchunk1Size (16 bytes for PCM/float)
                            3,  # AudioFormat (3 = IEEE Float)
                            num_channels,  # NumChannels
                            sample_rate,  # SampleRate
                            sample_rate * num_channels * 4,  # ByteRate
                            num_channels * 4,  # BlockAlign (4 bytes/channel)
                            32  # BitsPerSample
                            ))

        # data Subchunk
        f.write(struct.pack('<4sI', b'data', subchunk2_size))
        # Write float samples (little-endian)
        f.write(samples.tobytes(order='C'))  # np.float32.tobytes uses little-endian by default