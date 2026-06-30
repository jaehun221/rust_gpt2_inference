use ndarray::{ Array1, Array2, Axis, s };
use crate::{ config::Config, weights::Weights, operation::{attention, layer_norm, mlp} };


pub struct Gpt2 {
    wte: Array2<f32>,
    wpe: Array2<f32>,
    block: Vec<Block>,
    ln_f: Ln,
}

impl Gpt2 {
    pub fn from_weight(w: &Weights, config: &Config) -> Gpt2 {
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

    pub fn generate(&self, input: &Vec<usize>, config: &Config, caches: &mut Vec<KVCache>) -> usize {
        let wte = &self.wte;
        let wpe = &self.wpe;

        let input_len = input.len();

        let token = wte.select(Axis(0), &input);
        let position = wpe.slice(s![caches[0].k.nrows()..caches[0].k.nrows() + input_len, ..]);

        let mut embd = token + position;
        
        for i in 0..config.n_layer {
            let w = &self.block[i];
            let ln1 = layer_norm(&embd, &w.ln1.weight, &w.ln1.bias, config);
            embd = embd + attention(&ln1, &w.attn, config, &mut caches[i]);
            let ln2 = layer_norm(&embd, &w.ln2.weight, &w.ln2.bias, config);
            embd = embd + mlp(&ln2, &w.mlp);
            
        };

        let embd = layer_norm(&embd, &self.ln_f.weight, &self.ln_f.bias, config);

        let logits = embd.dot(&wte.t());

        let last_logit = logits.row(logits.nrows() -1);

        let next_token = last_logit
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        next_token
    
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

pub struct Ln {
    weight: Array1<f32>,
    bias: Array1<f32>,
}

pub struct Mlp {
    pub c_fc: Linear,
    pub c_proj: Linear,
}

pub struct Attn {
    pub c_attn: Linear,
    pub c_proj: Linear,
}

pub struct KVCache {
    pub k: Array2<f32>,
    pub v: Array2<f32>,
}

impl KVCache {
    pub fn cache_set(config: &Config) -> Vec<KVCache> {
        let caches: Vec<KVCache> = (0..config.n_layer)
                .map(|_| KVCache {
                k: Array2::zeros((0, config.n_embd)),
                v: Array2::zeros((0, config.n_embd)),
            })
            .collect();

        caches
    }
}