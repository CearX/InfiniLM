use tensor::*;

// pub fn layer_norm(x: &Tensor<T>) -> Tensor<T> {
//     todo!()
// }

// pub fn gelu(x: &Tensor<T>) -> Tensor<T> {
//     todo!()
// }

#[test]
fn encoder_forward() {
    // tensors to operators
    // operators: tensor input and tensor output


}

#[test]
fn forward() {
    encoder_forward();
    // layer_norm_forward();
    // attn_norm_forward();
    // attn_qkv_forward();
    // attn_o_forward();
    // ffn_norm_forward();
    // ffn_gate_up_forward();
    // ffn_down_forward();
    // output_norm_forward();
    // output_forward();
}
// worker.launch; tensor: x, x1;

#[test]
fn parametertensor_test() {
    pub struct Parametertensor {
        token_embd: Tensor<usize>,              // wte
        position_embd: Tensor<usize>,           // wpe
        
        // 12 x block
        // blcks: Box<BlkStorage<Tensor<usize>>>,
        blcks: Vec<Weights>,
       
        output_norm_bias: Tensor<usize>,        // lnfb
        output_norm_weight: Tensor<usize>,      // lnfw
        output: Tensor<usize>,
    }
    
    pub struct Weights {
        // attn
        pub attn_qkv_bias: Tensor<usize>,       // qkvb
        pub attn_qkv_weight: Tensor<usize>,     // qkvw
        pub attn_output_bias: Tensor<usize>,    // attprojb
        pub attn_output_weight: Tensor<usize>,  // attprojw
        // ln1
        pub attn_norm_bias: Tensor<usize>,      // ln1w
        pub attn_norm_weight: Tensor<usize>,    // ln1b

        // MLP
        pub ffn_up_bias: Tensor<usize>,         // fcb
        pub ffn_up_weight: Tensor<usize>,       // fcw
        pub ffn_down_bias: Tensor<usize>,       // fcprojb
        pub ffn_down_weight: Tensor<usize>,     // fcprojw
        // ln2
        pub ffn_norm_bias: Tensor<usize>,       // ln2w
        pub ffn_norm_weight: Tensor<usize>,     // ln2b
    }

// pub struct  Storage<T> {
//     pub meta: Gpt2Meta,
//     pub blocks: Box<[BlkStorage<T>]>,
//     pub output_norm_bias: T,
//     pub output_norm_weight: T,
//     pub position_embd_weight: T,
//     pub token_embd_weight: T,
//     pub output_weight: T,
// }
// #[derive(Clone, Copy)]
// pub struct BlkStorage<T> {
//     pub attn_qkv_bias: T,
//     pub attn_qkv_weight: T,
//     pub attn_output_bias: T,
//     pub attn_output_weight: T,
//     pub attn_norm_bias: T,
//     pub attn_norm_weight: T,

//     pub ffn_up_bias: T,
//     pub ffn_up_weight: T,
//     pub ffn_down_bias: T,
//     pub ffn_down_weight: T,
//     pub ffn_norm_bias: T,
//     pub ffn_norm_weight: T,
// }
}

#[test]
fn activationtensors2(){

}