import os
import sys

import numpy as np
import torch
import tvm_ffi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import cuda2np as c2n

DIR = os.path.dirname(__file__)
mod = tvm_ffi.load_module(f"{DIR}/../build/libcuda2np-trt.so")

np.random.seed(43)  # For reproducibility


def test_clip(
    numel: int,
    clip_min: float,
    clip_max: float,
):
    # Input tensor
    input_np = np.random.uniform(-10.0, 10.0, size=numel).astype(np.float32)

    # Output tensor (pre-allocated)
    output_np = np.zeros_like(input_np)

    # NumPy result
    c2n.trt.clip(input_np.copy(), output_np, clip_min=clip_min, clip_max=clip_max)

    # CUDA result
    input_cuda = torch.from_numpy(input_np).to("cuda")
    output_cuda = torch.zeros_like(input_cuda)

    mod.clip(input_cuda, output_cuda, clip_min, clip_max)

    status = np.allclose(output_cuda.cpu().numpy(), output_np)
    if not status:
        print(f"Test failed for numel={numel}, clip_min={clip_min}, clip_max={clip_max}")
        print(f"Expected: {output_np}")
        print(f"Got: {output_cuda.cpu().numpy()}")
    else:
        print(f"Test passed for numel={numel}, clip_min={clip_min}, clip_max={clip_max}")


if __name__ == "__main__":
    test_clip(numel=16, clip_min=-2.0, clip_max=2.0)
