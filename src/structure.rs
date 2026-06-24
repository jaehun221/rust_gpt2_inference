use ndarray::{ Array1, Array2 };


pub struct Linear {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

pub struct Embd {
    wte: Array2<f32>,
    wpe: Array2<f32>,
}

pub struct Ln {
    weight: Array1<f32>,
    bias: Array1<f32>,
}

pub struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

pub struct Attn {
    c_attn: Linear,
    c_proj: Linear,
    attn_bias: Array2<f32>,
}

pub struct Gpt2 {
    ln1: Ln,
    attn: Attn,
    ln2: Ln,
    mlp: Mlp,
}