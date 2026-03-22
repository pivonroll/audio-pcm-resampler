from enum import Enum
from typing import Union

import numpy as np
import soxr
from numpy.typing import NDArray


SoxDataType = Union[NDArray[np.int8], NDArray[np.int16], NDArray[np.int32], NDArray[np.float32], NDArray[np.float64]]

class ResampleQuality(str, Enum):
    QQ = "QQ"
    LowQuality = "LQ"
    MediumQuality = "MQ"
    HighQuality = "HQ"
    VeryHighQuality = "VHQ"

def resample_audio(input_signal: SoxDataType,
                   input_sample_rate: int,
                   output_sample_rate: int,
                   quality: ResampleQuality = ResampleQuality.HighQuality) -> SoxDataType:
    """Resample audio signal from input_rate to output_rate using soxr."""
    input_bitrate = input_signal.dtype
    if input_bitrate == np.int8:
        # soxr does not support int8, convert to int16 first
        input_signal = input_signal.astype(np.int16)

    resampled_signal = soxr.resample(input_signal, input_sample_rate, output_sample_rate, quality=quality.value)

    # Convert back to original dtype if needed
    if resampled_signal.dtype != input_bitrate:
        resampled_signal = resampled_signal.astype(input_bitrate)

    return resampled_signal