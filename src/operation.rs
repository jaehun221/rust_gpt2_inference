use ndarray::{ Array1, Array2, Axis, s };
use crate::{ config::Config, structure::{Attn, KVCache, Mlp} };


pub fn attention(input: &Array2<f32>, w: &Attn, config: &Config, cache: &mut KVCache) -> Array2<f32> {
    let n_embd = config.n_embd;
    let n_head = config.n_head;

    let qkv = input.dot(&w.c_attn.weight) + &w.c_attn.bias;

    let q = qkv.slice(s![.., ..n_embd]);
    let k = qkv.slice(s![.., n_embd..n_embd*2]);
    let v = qkv.slice(s![.., n_embd*2..]);

    cache.k = ndarray::concatenate(Axis(0), &[cache.k.view(), k.view()])
        .expect("k concat failed");
    cache.v = ndarray::concatenate(Axis(0), &[cache.v.view(), v.view()])
        .expect("v concat failed");


    let k = &cache.k;
    let v = &cache.v;

    let head_dim = n_embd / n_head;
    
    let mut attn_score = Array2::zeros((input.nrows(), n_embd));

    

    for i in 0..n_head {
        let q_h = q.slice(s![.., i*head_dim..(i+1)*head_dim]);
        let k_h = k.slice(s![.., i*head_dim..(i+1)*head_dim]);
        let v_h = v.slice(s![.., i*head_dim..(i+1)*head_dim]);

        let qk_h = q_h.dot(&k_h.t()) / (head_dim as f32).sqrt();

        let masked_qk_h = qk_h;

        let attn_weight = softmax(&masked_qk_h).dot(&v_h);
        attn_score.slice_mut(s![.., i*head_dim..(i+1)*head_dim]).assign(&attn_weight);

    }
    
    
    attn_score.dot(&w.c_proj.weight) + &w.c_proj.bias
    
}

pub fn layer_norm(input: &Array2<f32>, weight: &Array1<f32>, bias: &Array1<f32>, config: &Config) -> Array2<f32> {
    let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let var = input.var_axis(Axis(1), 0.).insert_axis(Axis(1));
    let eps = config.layer_norm_epsilon;

    let output = (input-&mean) / (var+eps).mapv(|x| x.sqrt());

    (output * weight) + bias
}

pub fn softmax(qk: &Array2<f32>) -> Array2<f32> {
    let max_value = qk.clone().fold_axis(Axis(1), f32::NEG_INFINITY, |&acc, &x| acc.max(x)); // 각 행의 가장 큰 값만 남김
    let qk = (qk - max_value.insert_axis(Axis(1))).mapv(|x| x.exp());
    let sum = qk.sum_axis(Axis(1)).insert_axis(Axis(1));

    qk / sum
}

pub fn gelu(input: &Array2<f32>) -> Array2<f32> {

    let pi = (2.0 / std::f32::consts::PI).sqrt();
    let num: f32 = 0.044715;

    input.mapv(|x| 0.5 * x * (1.0 + (pi * (x + num * x * x * x)).tanh()))
}


pub fn mlp(input: &Array2<f32>, w: &Mlp) -> Array2<f32> {
    gelu(&(input.dot(&w.c_fc.weight) + &w.c_fc.bias)).dot(&w.c_proj.weight) + &w.c_proj.bias
}
