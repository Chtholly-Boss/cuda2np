#include "common.cuh"
#include <cuda_fp16.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace plugin2numpy {

template <typename T1, typename T2, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__ void clipKernel(int n, const T1 clipMin, const T1 clipMax, const T2 *input, T2 *output) {
    int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
    if (i < n) {
        T1 inputElement = static_cast<T1>(input[i]);
        T1 tmp          = inputElement > clipMin ? inputElement : clipMin;
        output[i]       = static_cast<T2>(tmp < clipMax ? tmp : clipMax);
    }
}

void clip(
    // Input tensor
    tvm::ffi::TensorView input,
    // Output tensor
    tvm::ffi::TensorView output,
    // Parameters
    float clip_min,
    float clip_max) {
    int64_t numel    = input.shape().Product();
    DLDataType dtype = input.dtype();

    const int BS     = 512;
    const int GS     = divUp(numel, BS);

    // Dispatch based on dtype
    if (dtype.code == kDLFloat && dtype.bits == 32) {
        clipKernel<float, float, BS>
            <<<GS, BS>>>(numel, clip_min, clip_max, static_cast<const float *>(input.data_ptr()), static_cast<float *>(output.data_ptr()));
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
        clipKernel<float, half, BS><<<GS, BS>>>(numel, clip_min, clip_max, static_cast<const half *>(input.data_ptr()), static_cast<half *>(output.data_ptr()));
    } else if (dtype.code == kDLInt && dtype.bits == 32) {
        clipKernel<int32_t, int32_t, BS><<<GS, BS>>>(numel,
                                                     static_cast<int32_t>(clip_min),
                                                     static_cast<int32_t>(clip_max),
                                                     static_cast<const int32_t *>(input.data_ptr()),
                                                     static_cast<int32_t *>(output.data_ptr()));
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
        clipKernel<int8_t, int8_t, BS><<<GS, BS>>>(numel,
                                                   static_cast<int8_t>(clip_min),
                                                   static_cast<int8_t>(clip_max),
                                                   static_cast<const int8_t *>(input.data_ptr()),
                                                   static_cast<int8_t *>(output.data_ptr()));
    } else {
        TVM_FFI_THROW(TypeError) << "Unsupported dtype for clip: code=" << dtype.code << ", bits=" << dtype.bits;
    }
}

} // namespace plugin2numpy

TVM_FFI_DLL_EXPORT_TYPED_FUNC(clip, plugin2numpy::clip)
