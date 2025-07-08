template <class Tdata>
static __device__ void kernel(
    Tdata *__restrict__ y,
    int const sny,
    int const smy,
    int const shy,
    int const swy,
    Tdata const *__restrict__ x,
    int const snx,
    int const smx,
    int const shx,
    int const swx,
    Tdata const *__restrict__ b,
    int const snb,
    int const smb,
    int const shb,
    int const swb) {
    auto in = blockIdx.x,
         im = blockIdx.y,
         ih = blockIdx.z,
         iw = threadIdx.x,
         iy = in * sny + im * smy + ih * shy + iw * swy,
         ix = in * snx + im * smx + ih * shx + iw * swx,
         ib = in * snb + im * smb + ih * shb + iw * swb;

    y[iy] = Tdata(x[ix] + b[ib]);
}
