from typing import Union, Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from audio_pcm_resampler.audio_format import AudioFormats, PCMFormat, ULAWFormat
from audio_pcm_resampler.resampler import resample_audio
from audio_pcm_resampler.data_type_conversion import get_bit_rate_converter
import audioop


class AudioResamplerConfig(BaseModel):
    input_format: AudioFormats
    output_format: AudioFormats


class AudioResampler:
    def __init__(self, config: AudioResamplerConfig):
        self.config = config
        self._convert = self._get_conversion_function()

    def pcm_to_pcm(self, data: NDArray) -> NDArray:
        # convert sample rate
        _data = data
        if isinstance(data, bytes):
            _data = np.frombuffer(data, dtype=self.config.input_format.bit_depth.to_numpy())

        if self.config.input_format.sample_rate != self.config.output_format.sample_rate:
            _data = resample_audio(_data, self.config.input_format.sample_rate,
                                   self.config.output_format.sample_rate)

        bit_rate_converter = get_bit_rate_converter(
            self.config.input_format.bit_depth.to_numpy(),
            self.config.output_format.bit_depth.to_numpy()
        )
        return bit_rate_converter(_data)

    def ulaw_to_pcm(self, data: bytes) -> NDArray:
        # Here you would implement the actual u-law to PCM conversion
        pcm_bytes = audioop.ulaw2lin(data, self.config.output_format.bit_depth.byte_size())

        pcm_converter = get_bit_rate_converter(
            np.int8,
            self.config.output_format.bit_depth.to_numpy()
        )

        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        _data = pcm_converter(pcm_data)

        if self.config.output_format.sample_rate != 8000:
            _data = resample_audio(_data, self.config.input_format.sample_rate,
                                   self.config.output_format.sample_rate)
        return _data

    def pcm_to_ulaw(self, data: NDArray) -> bytes:
        _data = data
        if self.config.input_format.sample_rate != 8000:
            _data = resample_audio(_data, self.config.input_format.sample_rate, 8000)

        pcm_converter = get_bit_rate_converter(
            self.config.input_format.bit_depth.to_numpy(),
            np.int8
        )

        pcm_data = pcm_converter(_data)
        pcm_bytes = pcm_data.tobytes()
        # Here you would implement the actual PCM to u-law conversion
        ulaw_bytes = audioop.lin2ulaw(pcm_bytes, self.config.input_format.bit_depth.byte_size())
        return ulaw_bytes

    def process(self, data: Union[NDArray, bytes]) -> Union[NDArray, bytes]:
        return self._convert(data)

    def _get_conversion_function(self) -> Callable[[Union[NDArray, bytes]], Union[NDArray, bytes]]:
        if (isinstance(self.config.input_format, PCMFormat) and
                isinstance(self.config.output_format, PCMFormat)):
            return self.pcm_to_pcm
        elif (isinstance(self.config.input_format, ULAWFormat) and
              isinstance(self.config.output_format, PCMFormat)):
            return self.ulaw_to_pcm
        elif (isinstance(self.config.input_format, PCMFormat) and
              isinstance(self.config.output_format, ULAWFormat)):
            return self.pcm_to_ulaw
        else:
            raise NotImplementedError(f"Conversion from {type(self.config.input_format)} "
                                      f"to {type(self.config.output_format)} is not implemented.")
