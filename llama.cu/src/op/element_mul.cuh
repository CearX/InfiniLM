template <typename T>
__device__ inline void kernel(
    T* __restrict__ y,
    int const stride_y_n,
    T const* __restrict__ a,
    int const stride_a_n,
    T const* __restrict__ b,
    int const stride_b_n,
    int const n,
    int const d) {
    int n_idx = blockIdx.x;
    int d_blk = blockIdx.y;
    int d_idx = threadIdx.x + d_blk * blockDim.x;
    if (d_idx >= d) return;
    T const* a_row = a + n_idx * stride_a_n;
    T const* b_row = b + n_idx * stride_b_n;
    T* y_row = y + n_idx * stride_y_n;
    y_row[d_idx] = a_row[d_idx] * b_row[d_idx];
}
