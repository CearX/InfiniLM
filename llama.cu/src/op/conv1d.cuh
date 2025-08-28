// 统一的 conv1d kernel：同时处理状态更新和卷积计算
// 支持 prefill 和 decode 模式
// x: [n, d], state: [d, K], w: [d, k], b: [d], y: [n, d]
template <class Tdata, class Twb>
static __device__ void kernel_unified(
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
    float* __restrict__ state,
    int const s_state_c,
    int const s_state_k,
    int const n,
    int const d,
    int const kernel_size,
    int const padding) {

    const int c = blockIdx.x;
    const int pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (c >= d) return;

    // 首先更新状态（每个channel只需要一个thread处理）
    if (pos == 0) {
        float* row = state + c * s_state_c;

        if (n == 1) {
            // decode 模式：滑动窗口更新
            for (int j = 0; j < kernel_size - 1; ++j) {
                row[j * s_state_k] = row[(j + 1) * s_state_k];
            }
            row[(kernel_size - 1) * s_state_k] = static_cast<float>(x[c * s_d_x]);
        } else {
            // prefill 模式：写入序列的最后 K 步
            const int pad = (kernel_size > n) ? (kernel_size - n) : 0;
            // 前 pad 个位置写 0
            for (int j = 0; j < pad; ++j) {
                row[j * s_state_k] = 0.f;
            }
            // 剩余位置拷贝最后 min(n,k) 步
            const int copy = (kernel_size - pad);
            for (int j = 0; j < copy; ++j) {
                const int t = n - copy + j; // 对应 x 的时间步索引
                const int xi = t * s_n_x + c * s_d_x;
                row[(pad + j) * s_state_k] = static_cast<float>(x[xi]);
            }
        }
    }

    // 同步确保状态更新完成
    __syncthreads();

    // 计算卷积输出
    const int seq_len = gridDim.y * blockDim.x;
    if (pos >= seq_len) return;

    if (n == 1) {
        // decode 模式：使用状态计算输出
        if (pos == 0) {  // decode时只有一个输出
            Tdata sum = Tdata(0);
            float* row = state + c * s_state_c;
            for (int k = 0; k < kernel_size; ++k) {
                const int w_pos = c * s_d_w + k * s_k_w;
                sum += Tdata(w[w_pos]) * Tdata(row[k * s_state_k]);
            }
            sum += Tdata(b[c * s_d_b]);
            y[c * s_d_y] = sum;
        }
    } else {
        // prefill 模式：标准卷积计算
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
}
