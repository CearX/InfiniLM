use crate::memory::MemPages;
use cuda::{VirByte, VirMem};
use nn::{Tensor, digit_layout::types};

/// 每层的 Mamba 状态
pub struct MambaLayerState {
    conv: Tensor<VirMem, 2>, // [d_inner, d_conv]
    ssm: Tensor<VirMem, 2>,  // [d_inner, d_state]
    conv_mapped: usize,
    ssm_mapped: usize,
}

pub struct MambaCache {
    pub layers: Box<[MambaLayerState]>,
    pub conv_size_per_layer: usize,
    pub ssm_size_per_layer: usize,
}

impl MambaCache {
    pub fn new(
        n_layers: usize,
        d_inner: usize,
        d_conv: usize,
        d_state: usize,
        pages: &mut MemPages,
    ) -> Self {
        let conv_size_per_layer = d_inner * d_conv * 4; // F32
        let ssm_size_per_layer = d_inner * d_state * 4; // F32

        let layers = (0..n_layers)
            .map(|_| {
                let conv_tensor = Tensor::from_dim_slice(types::F32, [d_inner, d_conv])
                    .map(|len| pages.reserve_vir(len));

                let ssm_tensor = Tensor::from_dim_slice(types::F32, [d_inner, d_state])
                    .map(|len| pages.reserve_vir(len));

                MambaLayerState {
                    conv: conv_tensor,
                    ssm: ssm_tensor,
                    conv_mapped: 0,
                    ssm_mapped: 0,
                }
            })
            .collect();

        let mut cache = Self {
            layers,
            conv_size_per_layer,
            ssm_size_per_layer,
        };

        // 立即映射所有层的物理页
        for layer_idx in 0..n_layers {
            cache.ensure_mapped(layer_idx, pages);
        }

        cache
    }

    /// 更新 conv cache 的物理页映射
    pub fn update_conv_mapping(&mut self, layer_idx: usize, pages: &mut MemPages) {
        let layer = &mut self.layers[layer_idx];
        let page_size = pages.page_size();
        let target = self.conv_size_per_layer.div_ceil(page_size);

        let mem = layer.conv.get_mut();
        use std::cmp::Ordering::{Equal, Greater, Less};
        match layer.conv_mapped.cmp(&target) {
            Less => pages.map(mem, layer.conv_mapped..target),
            Greater => pages.unmap(mem, target..layer.conv_mapped),
            Equal => {}
        }
        layer.conv_mapped = target;
    }

    /// 更新 ssm cache 的物理页映射
    pub fn update_ssm_mapping(&mut self, layer_idx: usize, pages: &mut MemPages) {
        let layer = &mut self.layers[layer_idx];
        let page_size = pages.page_size();
        let target = self.ssm_size_per_layer.div_ceil(page_size);

        let mem = layer.ssm.get_mut();
        use std::cmp::Ordering::{Equal, Greater, Less};
        match layer.ssm_mapped.cmp(&target) {
            Less => pages.map(mem, layer.ssm_mapped..target),
            Greater => pages.unmap(mem, target..layer.ssm_mapped),
            Equal => {}
        }
        layer.ssm_mapped = target;
    }

    /// 获取 conv 状态的虚拟地址张量
    pub fn conv_tensor(&self, layer_idx: usize) -> Tensor<*const VirByte, 2> {
        self.layers[layer_idx].conv.as_ref().map(|vir| vir.as_ptr())
    }

    /// 获取 ssm 状态的虚拟地址张量
    pub fn ssm_tensor(&self, layer_idx: usize) -> Tensor<*const VirByte, 2> {
        self.layers[layer_idx].ssm.as_ref().map(|vir| vir.as_ptr())
    }

    /// 确保指定层的物理页已映射
    pub fn ensure_mapped(&mut self, layer_idx: usize, pages: &mut MemPages) {
        self.update_conv_mapping(layer_idx, pages);
        self.update_ssm_mapping(layer_idx, pages);
    }
}
