use ndarray::{ Array1, Array2, Axis, s };
use crate::{ config::Config, weights::Weights, operation::{attention, layer_norm, softmax, mlp} };


pub struct Gpt2 {
    wte: Array2<f32>,
    wpe: Array2<f32>,
    block: Vec<Block>,
    ln_f: Ln,
}

impl Gpt2 {
    fn from_weight(w: &Weights, config: Config) -> Gpt2 {
        let mut blocks: Vec<Block> = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(
                Block {
                    ln1: Ln { weight: w.get_tensor1(&format!("h.{i}.ln_1.weight")), bias: w.get_tensor1(&format!("h.{i}.ln_1.bias"))},
                    attn: Attn {
                        c_attn: Linear { weight: w.get_tensor2(&format!("h.{i}.attn.c_attn.weight")), bias: w.get_tensor1(&format!("h.{i}.attn.c_attn.bias")) },
                        c_proj: Linear { weight: w.get_tensor2(&format!("h.{i}.attn.c_proj.weight")), bias: w.get_tensor1(&format!("h.{i}.attn.c_proj.bias")) },
                    },
                    ln2: Ln { weight: w.get_tensor1(&format!("h.{i}.ln_2.weight")), bias: w.get_tensor1(&format!("h.{i}.ln_2.bias")) },
                    mlp: Mlp {
                        c_fc: Linear { weight: w.get_tensor2(&format!("h.{i}.mlp.c_fc.weight")), bias: w.get_tensor1(&format!("h.{i}.mlp.c_fc.bias")) },
                        c_proj: Linear { weight: w.get_tensor2(&format!("h.{i}.mlp.c_proj.weight")), bias: w.get_tensor1(&format!("h.{i}.mlp.c_proj.bias")) },
                    },
                }
            )
        }

        Gpt2 { 
            wte: w.get_tensor2("wte.weight"), 
            wpe: w.get_tensor2("wpe.weight"),
            block: blocks,
            ln_f: Ln {
                weight: w.get_tensor1("ln_f.weight"),
                bias: w.get_tensor1("ln_f.bias"),
            }
        }

    }

    fn generate(&self, input: &Vec<usize>, config: &Config) -> Vec<usize> {
        let wte = &self.wte;
        let wpe = &self.wpe;

        let input_len = input.len();

        let token = wte.select(Axis(0), &input);
        let position = wpe.slice(s![0..input_len, ..]);

        let embd = token + position;
        
        let weight = &self.block;
        for i in 0..config.n_layer {
            layer_norm(input, weight, bias, config)
        };
        
    } 
}

pub struct Block {
    ln1: Ln,
    attn: Attn,
    ln2: Ln,
    mlp: Mlp,
}

pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
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
    pub c_attn: Linear,
    pub c_proj: Linear,
}

