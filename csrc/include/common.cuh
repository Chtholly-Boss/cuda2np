#pragma once

inline __host__ __device__ constexpr auto divUp(auto x, auto n) { return (x + n - 1) / n; }
