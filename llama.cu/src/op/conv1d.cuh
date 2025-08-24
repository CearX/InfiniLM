// prefill: 将 full-seq 的最后 K 步写入 conv state，左侧零填充到 K
// x: [n, d], state: [d, K]
template <class Tdata>
static __device__ void kernel_write_state(
    Tdata const* __restrict__ x,
    int const s_n_x,
    int const s_d_x,
    float* __restrict__ state,
    int const s_state_c,
    int const s_state_k,
    int const n,
    int const d,
    int const k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;

    float* row = state + c * s_state_c;
    const int pad = (k > n) ? (k - n) : 0;
    // 前 pad 个位置写 0
    for (int j = 0; j < pad; ++j) {
        row[j * s_state_k] = 0.f;
    }
    // 剩余位置拷贝最后 min(n,k) 步
    const int copy = (k - pad);
    for (int j = 0; j < copy; ++j) {
        const int t = n - copy + j; // 对应 x 的时间步索引
        const int xi = t * s_n_x + c * s_d_x;
        row[(pad + j) * s_state_k] = static_cast<float>(x[xi]);
    }
}

//  x[n, d], groups=d, w[d, k], b[d]
template <class Tdata, class Twb>
static __device__ void kernel(
    Tdata* __restrict__ y,
    int const s_n_y,
    int const s_d_y,
    Tdata const* __restrict__ x,
    int const s_n_x,
    int const s_d_x,
    Twb const* __restrict__ w,
    int const s_d_w,
    int const s_k_w,
    Twb const* __restrict__ b,
    int const s_d_b,
    int const kernel_size,
    int const padding) {

    const int c = blockIdx.x;
    const int pos = blockIdx.y * blockDim.x + threadIdx.x;

    const int seq_len = gridDim.y * blockDim.x;
    if (pos >= seq_len) return;

    Tdata sum = Tdata(0);
    for (int k = 0; k < kernel_size; ++k) {
        const int l_in = pos + k - padding;
        if (l_in >= 0 && l_in < seq_len) {
            const int x_pos = l_in * s_n_x + c * s_d_x;
            const int w_pos = c * s_d_w + k * s_k_w;
            sum += Tdata(x[x_pos]) * Tdata(w[w_pos]);
        }
    }

    sum += Tdata(b[c * s_d_b]);

    const int y_pos = pos * s_n_y + c * s_d_y;
    y[y_pos] = sum;
}

// decode: 输入 x_t[n,d] (n = seq = 1)，就地读写 state [d, K]，并计算 y[n, d]
template <class Tdata, class Twb>
static __device__ void kernel_step(
    Tdata* __restrict__ y,
    int const s_n_y,
    int const s_d_y,
    Tdata const* __restrict__ x_t,
    int const s_n_x,
    int const s_d_x,
    Twb const* __restrict__ w,
    int const s_d_w,
    int const s_k_w,
    Twb const* __restrict__ b,
    int const s_d_b,
    float* __restrict__ state,
    int const s_state_c,
    int const s_state_k,
    int const kernel_size,
    int const k_minus_1,
    int const d_channels)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_channels) return;

    // 更新状态
    float* row = state + c * s_state_c;
    for (int j = 0; j < kernel_size - 1; ++j) {
        row[j * s_state_k] = row[(j + 1) * s_state_k];
    }
    row[(kernel_size - 1) * s_state_k] = static_cast<float>(x_t[c * s_d_x]);
    // 计算输出
    Tdata sum = Tdata(0);
    for (int i = 0; i < kernel_size; ++i) {
        const int wi = c * s_d_w + i * s_k_w;
        const int si = c * s_state_c + i * s_state_k;
        sum += Tdata(w[wi]) * Tdata(state[si]);
    }
    sum += Tdata(b[c * s_d_b]);

    y[c * s_d_y] = sum;
}
