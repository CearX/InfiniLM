mod chat_template;
pub(crate) mod image;
mod llama;
pub(crate) mod qw2vl_mmproj;

use crate::utils::{Blob, Data};
use ggus::{
    GENERAL_ALIGNMENT, GGuf, GGufError, GGufFileName, GGufMetaDataValueType, GGufMetaKV,
    GGufMetaMap,
};
use memmap2::Mmap;
use nn::{Tensor, digit_layout::types};
use std::{collections::HashMap, fmt::Debug, fs::File, path::Path, thread};

pub(crate) use chat_template::ChatTemplate;

pub use chat_template::Message;

/// GGuf 模型，可能来自多个分片文件。
pub(crate) struct GGufModel<'a> {
    /// 元数据键值对。
    pub meta_kvs: HashMap<&'a str, GGufMetaKV<'a>>,
    /// 张量。
    pub tensors: HashMap<&'a str, Tensor<Data<'a>, 2>>,
}

impl<'a> GGufModel<'a> {
    /// 从多个分片文件中读取 GGuf 模型。
    pub fn read(files: impl IntoIterator<Item = &'a [u8]> + 'a) -> Self {
        let mut ans = Self {
            meta_kvs: Default::default(),
            tensors: Default::default(),
        };
        thread::scope(|s| {
            for (i, thread) in files
                .into_iter()
                .map(|data| s.spawn(|| GGuf::new(data)))
                .collect::<Vec<_>>()
                .into_iter()
                .enumerate()
            {
                thread
                    .join()
                    .unwrap()
                    .and_then(|gguf| ans.merge(gguf))
                    .unwrap_or_else(|e| panic!("Error at file {i}: {e}"));
            }
        });
        ans
    }

    fn merge(&mut self, gguf: GGuf<'a>) -> Result<(), GGufError> {
        for (k, kv) in gguf.meta_kvs {
            if k == GENERAL_ALIGNMENT || k.starts_with("split.") {
                continue;
            }
            if self.meta_kvs.insert(k, kv).is_some() {
                return Err(GGufError::DuplicateMetaKey(k.into()));
            }
        }

        for (name, t) in gguf.tensors {
            use std::collections::hash_map::Entry::{Occupied, Vacant};
            match self.tensors.entry(name) {
                Occupied(_) => return Err(GGufError::DuplicateTensorName(name.into())),
                Vacant(vacant) => {
                    let t = t.to_info();
                    let ty = t.ty().to_digit_layout();
                    let shape = t
                        .shape()
                        .iter()
                        .rev()
                        .map(|&x| x as usize)
                        .collect::<Vec<_>>();
                    vacant.insert(Tensor::from_dim_slice(ty, &*shape).map(|len| {
                        assert_eq!(len, t.nbytes());
                        gguf.data[t.offset()..][..t.nbytes()].into()
                    }));
                }
            }
        }

        Ok(())
    }
}

impl GGufMetaMap for GGufModel<'_> {
    fn get(&self, key: &str) -> Option<(GGufMetaDataValueType, &[u8])> {
        self.meta_kvs.get(key).map(|kv| (kv.ty(), kv.value_bytes()))
    }
}

/// 从指定文件的路径出发，映射所有分片文件。
pub(crate) fn map_files(path: impl AsRef<Path>) -> Box<[Mmap]> {
    fn throw(path: &Path, e: impl Debug) -> ! {
        let path = path.display();
        panic!(
            "\
Error occurred at path: {path}
  error: {e:?}"
        )
    }

    #[inline]
    fn map_file(path: &Path) -> Mmap {
        let file = File::open(path).unwrap_or_else(|e| throw(path, e));
        unsafe { Mmap::map(&file) }.unwrap()
    }

    let path = path.as_ref();
    let name = GGufFileName::try_from(path).unwrap_or_else(|e| throw(path, e));

    if name.shard_count() == 1 {
        Box::new([map_file(path)])
    } else {
        let dir = path.parent().unwrap();
        name.iter_all()
            .map(|name| map_file(&dir.join(name.to_string())))
            .collect()
    }
}

/// 构造 sin cos 表张量
pub(crate) fn build_sin_cos<'a, const N: usize>(
    nctx: usize,
    dh: usize,
    theta: f32,
    mut pos_scaling: impl FnMut(usize, usize) -> f32,
) -> [Tensor<Data<'a>, N>; 2] {
    let d = dh / 2;
    let ty = types::F32;
    let mut sin = Blob::new(nctx * d * ty.nbytes());
    let mut cos = Blob::new(nctx * d * ty.nbytes());
    let theta = theta.powf(-(d as f32).recip());

    {
        let ([], sin, []) = (unsafe { sin.align_to_mut() }) else {
            unreachable!()
        };
        let ([], cos, []) = (unsafe { cos.align_to_mut() }) else {
            unreachable!()
        };
        for pos in 0..nctx {
            for i in 0..d {
                let (sin_, cos_) = (pos_scaling(pos, i) * theta.powi(i as _)).sin_cos();
                sin[pos * d + i] = sin_;
                cos[pos * d + i] = cos_;
            }
        }
    }

    let tensor = |data: Blob| {
        Tensor::from_dim_slice(ty, [nctx, d]).map(|len| {
            assert_eq!(len, data.len());
            data.into()
        })
    };
    [tensor(sin), tensor(cos)]
}
