template <class Tdata, class Tad>
static __device__ void kernel_with_writeback(
    Tdata *__restrict__ out,
    int const stride_out_seq,
    int const stride_out_d_in,
    Tdata const *__restrict__ u,
    int const stride_u_seq,
    int const stride_u_d_in,
    Tdata const *__restrict__ delta,
    int const stride_delta_seq,
    int const stride_delta_d_in,
    Tad const *__restrict__ A,
    int const stride_A_d_in,
    int const stride_A_n,
    Tdata const *__restrict__ B,
    int const stride_B_seq,
    int const stride_B_n,
    Tdata const *__restrict__ C,
    int const stride_C_seq,
    int const stride_C_n,
    Tad const *__restrict__ D,
    int const stride_D_d_in,
    int const seq_len,
    int const d_in,
    int const n,
    float *__restrict__ state_out,
    int const stride_state_d,
    int const stride_state_n)
{
    assert(gridDim.x == 5120 && gridDim.x == d_in);
    const int i_d_in = blockIdx.x + blockIdx.y * gridDim.x;
    assert(blockIdx.x >= 0 && blockIdx.x < 5120);
    assert(blockIdx.y == 0);
    assert(threadIdx.x == 0);
    if (i_d_in >= d_in) return;

    extern __shared__ unsigned char smem_raw[];
    float *x = reinterpret_cast<float *>(smem_raw);

    // 更新 cache
    if (seq_len > 1) {
        for (int i = 0; i < n; ++i)
            x[i] = 0.f;
    } else if (seq_len == 1) {
        for (int i = 0; i < n; ++i)
            x[i] = state_out[i_d_in * stride_state_d + i * stride_state_n];
    } else {
        assert(false);
    }

    for (int t = 0; t < seq_len; ++t) {
        const auto i_u = t * stride_u_seq + i_d_in * stride_u_d_in;
        const auto i_delta = t * stride_delta_seq + i_d_in * stride_delta_d_in;
        const auto i_B = t * stride_B_seq;
        const auto i_C = t * stride_C_seq;

        const Tdata u_ = u[i_u];
        const float dx = static_cast<float>(delta[i_delta]);
        const float delta_f = (dx <= 20.f) ? log1pf(expf(dx)) : dx;

        float y = 0.f;
        for (int i = 0; i < n; ++i)
        {
            const auto i_A = i_d_in * stride_A_d_in + i * stride_A_n;
            const auto i_B_ = i_B + i * stride_B_n;
            const float a = static_cast<float>(A[i_A]);
            const float b = static_cast<float>(B[i_B_]);
            float xi = x[i];
            const float u_f = static_cast<float>(u_);
            const float deltaA = expf(delta_f * a);
            const float deltaB_u = delta_f * b * u_f;
            // state equation
            xi = deltaA * xi + deltaB_u;
            x[i] = xi;

            const auto i_C_ = i_C + i * stride_C_n;
            // output equation
            y += static_cast<float>(C[i_C_]) * x[i];
        }

        const auto i_D = i_d_in * stride_D_d_in;
        y += static_cast<float>(D[i_D]) * static_cast<float>(u_);

        const auto i_out = t * stride_out_seq + i_d_in * stride_out_d_in;
        out[i_out] = static_cast<Tdata>(y);
    }

    // 写回 cache
    if (state_out != nullptr) {
        for (int i = 0; i < n; ++i) {
            state_out[i_d_in * stride_state_d + i * stride_state_n] = x[i];
        }
    }
}
