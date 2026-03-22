from enum import Enum
from typing import Union

from pydantic import BaseModel

from audio_pcm_resampler.data_type_conversion import NpBitDepth


class AudioFormat(str, Enum):
    PCM = "audio/pcm"
    ULAW = "audio/ulaw"
    ALAW = "audio/alaw"

class BitDepth(str, Enum):
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    def to_numpy(self) -> NpBitDepth:
        import numpy as np
        mapping = {
            BitDepth.INT8: np.int8,
            BitDepth.INT16: np.int16,
            BitDepth.INT32: np.int32,
            BitDepth.UINT8: np.uint8,
            BitDepth.UINT16: np.uint16,
            BitDepth.UINT32: np.uint32,
            BitDepth.FLOAT32: np.float32,
            BitDepth.FLOAT64: np.float64,
        }
        return mapping[self]

    def byte_size(self) -> int:
        mapping = {
            BitDepth.INT8: 1,
            BitDepth.INT16: 2,
            BitDepth.INT32: 4,
            BitDepth.UINT8: 1,
            BitDepth.UINT16: 2,
            BitDepth.UINT32: 4,
            BitDepth.FLOAT32: 4,
            BitDepth.FLOAT64: 8,
        }
        return mapping[self]


class PCMFormat(BaseModel):
    type: AudioFormat = AudioFormat.PCM
    sample_rate: int = 24000
    bit_depth: BitDepth = BitDepth.INT16

class ULAWFormat(BaseModel):
    type: AudioFormat = AudioFormat.ULAW
    sample_rate: int = 8000


AudioFormats = Union[PCMFormat, ULAWFormat]