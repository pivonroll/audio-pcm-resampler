from typing import List, Union, Callable, Type

import numpy as np

from numpy.typing import NDArray

SignedPCM = Union[NDArray[np.int8], NDArray[np.int16], NDArray[np.int32]]
UnsignedPCM = Union[NDArray[np.uint8], NDArray[np.uint16], NDArray[np.uint32]]
FloatPCM = Union[NDArray[np.float32], NDArray[np.float64]]
AllPCM = Union[SignedPCM, UnsignedPCM, FloatPCM]

SignedBitDepth = Type[Union[np.int8, np.int16, np.int32]]
UnsignedBitDepth = Type[Union[np.uint8, np.uint16, np.uint32]]
FloatBitDepth = Type[Union[np.float32, np.float64]]
NpBitDepth = Union[SignedBitDepth, FloatBitDepth]


def bytes_2_numpy(
        sound: bytes,
        target: Type[Union[np.int8, np.int16, np.int32, np.float32, np.float64]]
) -> Union[NDArray[np.int8], NDArray[np.int16], NDArray[np.int32], NDArray[np.float32], NDArray[np.float64]]:
    """Converts bytes to NumPy array of specified integer type (int8 or int16)."""
    return np.frombuffer(sound, dtype=target)


def int16_to_float32(sound: NDArray[np.int16]) -> NDArray[np.float32]:
    """Convert signed 16-bit NumPy array to float32 NumPy array normalized to [-1, 1]."""

    abs_max = np.abs(sound).max()
    float_sound = sound.astype('float32')

    # Min for int16 is -32768, max is 32767
    # So we normalize the sound to the range [-1, 1]
    # by dividing by 32768
    max_int_16 = np.abs(np.iinfo(np.int16).min)
    if abs_max > 0:
        float_sound *= 1/max_int_16
    float_sound = float_sound.squeeze()  # depends on the use case
    return float_sound


def pcm_to_float(
    input_data: List[int],
    target_dtype: Type[Union[np.float32, np.float64]] = np.float32
) -> Union[NDArray[np.float32], NDArray[np.float64]]:
    """Convert 8-bit or 16-bit PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

	  Args:
	    input_data : array_like
	        Input array, must have integral type.
	    target_dtype : data type, optional
	        Desired (floating point) data type.

    Returns:
	    numpy.ndarray
	        Normalized floating point data.

    """
    return pcm_to_float_numpy_array(np.asarray(input_data), target_dtype)


def pcm_to_float_numpy_array(
    input_data: Union[NDArray[np.int32], NDArray[np.int16], NDArray[np.int8]],
    target_dtype: Type[Union[np.float32, np.float64]] = np.float32
) -> Union[NDArray[np.float32], NDArray[np.float64]]:
    """Convert PCM signal to floating point normalized to [-1, 1] in 32-bit or 64-bit float.

    Use dtype='float32' for single precision.

    Args:
	    input_data : array_like
	        Input array, must have integral type.
	    target_dtype : data type, optional
	        Desired (floating point) data type.

    Returns:
	    numpy.ndarray
	        Normalized floating point data.

    """
    if input_data.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")

    target_dtype_info = np.iinfo(input_data.dtype)
    abs_max = 2 ** (target_dtype_info.bits - 1)
    offset = target_dtype_info.min + abs_max
    return (input_data.astype(target_dtype) - offset) / abs_max


def float2pcm(
    input_data: List[float],
    target_type: Type[Union[np.int16, np.int32]] = np.int16
) -> Union[NDArray[np.int16], NDArray[np.int32]]:
    """Convert floating point signal with a range [-1, 1] to signed 16-bit or 32-bit PCM.

    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.

    Args:
	    input_data : array_like
	        Input array, must have floating point type.
	    target_type : data type, optional
	        Desired (integer) data type.

    Returns:
	    numpy.ndarray
	        Integer data, scaled and clipped to the range of the given
	        *dtype*.
    """
    return float_to_pcm_numpy_array(np.asarray(input_data), target_type)


def float_to_pcm_numpy_array(
    input_data: NDArray[np.float32],
    target_type: Type[Union[np.int8, np.int16, np.int32]] = np.int16
) -> Union[NDArray[np.int8], NDArray[np.int16], NDArray[np.int32]]:
    """Convert floating point signal with a range [-1, 1] to signed 16-bit or 32-bit PCM.

    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.

    Args:
	    input_data : array_like
	        Input array, must have floating point type.
	    target_type : target data type, optional
	        Desired (integer) data type.

    Returns:
	    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.

    """
    if input_data.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    target_dtype = np.dtype(target_type)
    if target_dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    integer_info = np.iinfo(target_dtype)
    abs_max = np.abs(integer_info.min)
    offset = integer_info.min + abs_max
    return (input_data * abs_max + offset).clip(integer_info.min, integer_info.max).astype(target_dtype)


def pcm_24bit_to_32bit(data, channels=1, normalize=True):
    """Convert 24-bit PCM data to 32-bit.

    Args:
	    data : buffer
	        A buffer object where each group of 3 bytes represents one
	        little-endian 24-bit value.
	    channels : int, optional
	        Number of channels, by default 1.
	    normalize : bool, optional
	        If ``True`` (the default) the additional zero-byte is added as
	        least significant byte, effectively multiplying each value by
	        256, which leads to the maximum 24-bit value being mapped to the
	        maximum 32-bit value.  If ``False``, the zero-byte is added as
	        most significant byte and the values are not changed.

    Returns:
	    numpy.ndarray
        The content of *data* converted to an *int32* array, where each
        value was padded with zero-bits in the least significant byte
        (``normalize=True``) or in the most significant byte
        (``normalize=False``).

    """
    if len(data) % 3 != 0:
        raise ValueError('Size of data must be a multiple of 3 bytes')

    out = np.zeros(len(data) // 3, dtype='<i4')
    out.shape = -1, channels
    temp = out.view('uint8').reshape(-1, 4)
    if normalize:
        # write to last 3 columns, leave LSB at zero
        columns = slice(1, None)
    else:
        # write to first 3 columns, leave MSB at zero
        columns = slice(None, -1)
    temp[:, columns] = np.frombuffer(data, dtype='uint8').reshape(-1, 3)
    return out


def get_bit_rate_converter(
    source_bit_rate: NpBitDepth,
    target_bit_rate: NpBitDepth,
) -> Callable[[NDArray], NDArray]:
    """Get a function that converts PCM data from source to target bit rate.

    Args:
	    source_bit_rate : int
	        Source bit rate (np.int8, np.int16, np.int32, np.float32, or np.float64).
	    target_bit_rate : int
	        Target bit rate (np.int8, np.int16, np.int32, np.float32, or np.float64).

    Returns:
	    function
	        A function that takes a NumPy array of the source bit rate and
	        returns a NumPy array of the target bit rate.

    """
    if source_bit_rate == target_bit_rate:
        def identity(data: NDArray) -> NDArray:
            return data
        return identity

    if source_bit_rate in [np.int8, np.int16, np.int32] and target_bit_rate in [np.int8, np.int16, np.int32]:
        max_source = np.iinfo(source_bit_rate).max
        max_target = np.iinfo(target_bit_rate).max
        scale = max_target / max_source

        def convert(data: NDArray) -> NDArray:
            return (data * scale).astype(target_bit_rate)
        return convert

    if source_bit_rate in [np.float32, np.float64] and target_bit_rate in [np.float32, np.float64]:
        def convert(data: NDArray) -> NDArray:
            return data.astype(target_bit_rate)
        return convert

    if source_bit_rate in [np.int8, np.int16, np.int32] and target_bit_rate in [np.float32, np.float64]:
        def convert(data: NDArray) -> NDArray:
            return pcm_to_float_numpy_array(data, target_dtype=target_bit_rate)
        return convert

    if source_bit_rate in [np.float32, np.float64] and target_bit_rate in [np.int8, np.int16, np.int32]:
        def convert(data: NDArray) -> NDArray:
            return float_to_pcm_numpy_array(data, target_type=target_bit_rate)
        return convert

    raise ValueError("Unsupported bit rate conversion between "
                        f"{source_bit_rate} and {target_bit_rate}.")


def pcm_int_to_ulaw(pcm_data: SignedPCM) -> bytes:
    """Convert 16-bit PCM data to 8-bit u-law encoded data.

    Args:
        pcm_data : SignedPCM
            Input PCM data as a NumPy array of signed integers.
    Returns:
        bytes in u-law format.
    """
    import audioop

    pcm_bytes = pcm_data.tobytes()
    ulaw_bytes = audioop.lin2ulaw(pcm_bytes, np.iinfo(pcm_data.dtype).bits // 8)
    return ulaw_bytes


def ulaw_to_pcm_int(ulaw_data: bytes, target_type: SignedBitDepth) -> SignedPCM:
    """Convert 8-bit u-law encoded data to 16-bit PCM data.

    Args:
        ulaw_data : bytes
            Input u-law encoded data.
        target_type : Signed int PCM data type
    Returns:
        SignedPCM
            Output PCM data as a NumPy array of signed integers.
    """
    import audioop

    pcm_bytes = audioop.ulaw2lin(ulaw_data, np.iinfo(target_type).bits // 8)
    pcm_data = np.frombuffer(pcm_bytes, dtype=target_type)
    return pcm_data