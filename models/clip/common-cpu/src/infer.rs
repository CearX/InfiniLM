use crate::{Operators, Weights};
use clip::{ClipArgs, ClipMeta, ClipStorage, ClipWorker, Image, Tensor, D_POS_EMBD};
use gguf::{ggml_quants::digit_layout::types as ty, GGufModel};
use operators::{
    common_cpu::{Cpu, ThisThread},
    Blob,
};
use std::time::Instant;
use test_utils::Inference;

type Worker<'w> = ClipWorker<Operators, Weights<'w>>;

#[test]
fn test_infer() {
    let Some(Inference { model, .. }) = Inference::load() else {
        return;
    };
    let Some(picture) = test_utils::image() else {
        return;
    };

    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let storage = ClipStorage::from_gguf(&gguf);
    let meta = &storage.meta;
    println!("{meta:#?}");

    let &ClipMeta {
        dt,

        d_image,
        d_patch,

        image_mean,
        image_std,
        ..
    } = meta;

    let time = Instant::now();
    let image = Image::load(picture);
    println!("load image {:?}", time.elapsed());

    let time = Instant::now();
    let slices = image
        .slice_uhd(9, d_image, d_patch)
        .normalize(dt, image_mean, image_std);
    println!("slice image {:?}", time.elapsed());

    let weights = Weights::new(&storage);
    let mut worker = Worker::new(&Cpu, meta.clone(), weights);

    let whole = slices.whole();
    worker
        .launch(
            ClipArgs {
                raw: whole.to_nchw(),
                pos: pos70(1, whole.shape(), d_patch).map_slice(),
            },
            &mut [],
            &ThisThread,
        )
        .unwrap();

    if let Some(patches) = slices.patches_nchw() {
        let &[n, 3, h, w] = patches.shape() else {
            unreachable!()
        };
        worker
            .launch(
                ClipArgs {
                    raw: patches.map_slice(),
                    pos: pos70(n, [w, h], d_patch).map_slice(),
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
    }
}

fn pos70(n: usize, [w, h]: [usize; 2], d_patch: usize) -> Tensor<Blob> {
    let pos_w = w / d_patch;
    let pos_h = h / d_patch;

    let mut ans = Tensor::new(ty::U32, &[1, pos_w * pos_h])
        .broadcast(0, n)
        .map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<u32>() }) else {
        panic!()
    };

    for i in 0..pos_h * pos_w {
        let y = (i / pos_w) * D_POS_EMBD / pos_h;
        let x = (i % pos_w) * D_POS_EMBD / pos_w;
        data[i] = (y * D_POS_EMBD + x) as _;
    }

    ans
}

fn pos_resampler(n: usize, [w, h]: [usize; 2], d_patch: usize) -> Tensor<Blob> {
    let d = 3584;
    let pos_w = w / d_patch;
    let pos_h = h / d_patch;

    let mut ans = Tensor::new(ty::F32, &[1, pos_w * pos_h])
        .broadcast(0, n)
        .map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<f32>() }) else {
        panic!()
    };

    let pos_embed_t = get_2d_sincos_pos_embed(d, (pos_w, pos_h));

    for i in 0..pos_w * pos_h {
        for j in 0..d {
            data[i * d + j] = pos_embed_t[i][j];
        }
    }
    ans    
}


fn get_2d_sincos_pos_embed(embed_dim: usize, image_size: (usize, usize)) -> Vec<Vec<f32>> {
    let (grid_h_size, grid_w_size) = image_size;

    let mut grid_h: Vec<f32> = (0..grid_h_size).map(|i| i as f32).collect();
    let mut grid_w: Vec<f32> = (0..grid_w_size).map(|i| i as f32).collect();

    let mut grid: Vec<Vec<f32>> = vec![vec![0.0; grid_w_size]; grid_h_size];
    for h in 0..grid_h_size {
        for w in 0..grid_w_size {
            grid[h][w] = grid_w[w];
        }
    }

    let mut grid_2d: Vec<Vec<Vec<f32>>> = vec![grid.clone(), grid.clone()];
    for h in 0..grid_h_size {
        for w in 0..grid_w_size {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    let pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    let (H, W) = image_size;
    let mut pos_embed_2d: Vec<Vec<f32>> = vec![vec![0.0; embed_dim]; H * W];
    for h in 0..H {
        for w in 0..W {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w].clone();
        }
    }

    pos_embed_2d
}

fn get_2d_sincos_pos_embed_from_grid(embed_dim: usize, grid: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
    assert!(embed_dim % 2 == 0);

    let emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0].clone()); // (H, W, D/2)
    let emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1].clone()); // (H, W, D/2)

    let H = emb_h.len();
    let W = emb_h[0].len();
    let mut emb: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; embed_dim]; W]; H];

    for h in 0..H {
        for w in 0..W {
            for d in 0..(embed_dim / 2) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }

    emb
}

fn get_1d_sincos_pos_embed_from_grid_new(embed_dim: usize, pos: Vec<Vec<f32>>) -> Vec<Vec<Vec<f32>>> {
    assert!(embed_dim % 2 == 0);
    let H = pos.len();
    let W = pos[0].len();

    let mut omega: Vec<f32> = (0..embed_dim / 2)
        .map(|i| 1.0 / 10000.0f32.powi(i as i32 / (embed_dim / 2) as i32))
        .collect();

    let mut emb: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; embed_dim]; W]; H];
    for h in 0..H {
        for w in 0..W {
            for d in 0..(embed_dim / 2) {
                let out_value = pos[h][w] * omega[d];
                emb[h][w][d] = out_value.sin();
                emb[h][w][d + embed_dim / 2] = out_value.cos();
            }
        }
    }

    emb
}