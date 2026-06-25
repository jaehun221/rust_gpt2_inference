use ndarray::{ Array1, Array2 };
use crate::weights::Weights;


pub struct Gpt2 {
    ln1: Ln,
    attn: Attn,
    ln2: Ln,
    mlp: Mlp,
}

impl Gpt2 {
    fn generate(input: Vec<u32>) {
        let weights = Weights::weights_load("models/model.safetensors");
        let wte = Weights::get_tensor2(&weights, "wte.weight");

        let embd_wte: Array2<f32> = Array2::default((0, 0));
        let x = input.iter().map(|x| embd_wte.push_row(wte[x]));
    }  
}

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

