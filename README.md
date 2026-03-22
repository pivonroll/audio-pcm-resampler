# Audio Resampler

Audio Resampler is a Python library designed for converting audio data between different formats and sample rates. 
It supports PCM and u-law audio formats and provides a flexible API for audio processing.

## Features

- Convert audio data between PCM and u-law formats.
- Resample audio data to different sample rates.
- Support for various bit depths.
- Easy-to-use API with configuration options.

## Installation

To install the library, use pip:

```bash
pip install audio-pcm-resampler
```

## Usage

### Basic Example

Convert PCM format to PCM
```python
from audio_pcm_resampler.audio_resampler import AudioResampler, AudioResamplerConfig
from audio_pcm_resampler.audio_format import PCMFormat, ULAWFormat, BitDepth
from audio_pcm_resampler.data_type_conversion import bytes_2_numpy

# open input file
file = "I_want_to_create_the_world_int16_8khz.wav"
from audio_pcm_resampler.open_wav import open_wav_file
audio = open_wav_file(file)

# Initialize the resampler
resampler = AudioResampler(AudioResamplerConfig(
    # Input format must match the one from the file
    input_format=PCMFormat(
        sample_rate=audio.params.framerate,
        bit_depth=BitDepth.INT16,
    ),
    output_format=PCMFormat(
        sample_rate=24_000,
        bit_depth=BitDepth.INT16,
    )
))

# Process PCM input audio data, must use NDArray as input
input_data = bytes_2_numpy(audio.frames, resampler.config.input_format.bit_depth.to_numpy())

output_data = resampler.process(input_data)
```

Convert PCM to ULAW
```python
from audio_pcm_resampler.audio_resampler import AudioResampler, AudioResamplerConfig
from audio_pcm_resampler.audio_format import PCMFormat, ULAWFormat, BitDepth
from audio_pcm_resampler.data_type_conversion import bytes_2_numpy

# open input file
file = "I_want_to_create_the_world_int16_8khz.wav"
from audio_pcm_resampler.open_wav import open_wav_file
audio = open_wav_file(file)

# Initialize the resampler
resampler = AudioResampler(AudioResamplerConfig(
    # Input format must match the one from the file
    input_format=PCMFormat(
        sample_rate=audio.params.framerate,
        bit_depth=BitDepth.INT16,
    ),
    output_format=ULAWFormat()
))

# Process PCM input audio data, must use NDArray as input
input_data = bytes_2_numpy(audio.frames, resampler.config.input_format.bit_depth.to_numpy())

output_data = resampler.process(input_data)
```

## Requirements

- Python >= 3.10
- NumPy
- Pydantic
- soxr

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/pivonroll/audio-pcm-resampler).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author
 
[Radovan Zivkovic](mailto:ra.zivkovic@gmail.com)
