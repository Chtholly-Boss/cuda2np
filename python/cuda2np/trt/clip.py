import numpy as np


def clip(
    # Input tensor
    input: np.ndarray,  # shape: any shape, dtype: float32/float16/int32/int8
    # Output tensor
    output: np.ndarray,
    # Parameters
    clip_min: float,
    clip_max: float,
) -> None:
    """Clip input values to [clip_min, clip_max] range.

    Args:
        input: Input tensor to clip
        output: Output tensor (pre-allocated, filled in-place)
        clip_min: Minimum clipping bound
        clip_max: Maximum clipping bound
    """
    # Use NumPy's clip function
    np.clip(input, clip_min, clip_max, out=output)
