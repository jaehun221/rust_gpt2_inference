use ndarray::{ Array, Array1, Array2, Axis, s };
use crate::config::Config;


// pub fn Attention(input: &Array2<f32>, weight: &Array2<f32>, bias: &Array1<f32>, n_embd: &Config) -> Array2<f32> {
    
//     let n_embd = n_embd.n_embd;
//     let qkv = input.dot(weight) + bias;
//     let qk = &qkv.slice(s![.., ..n_embd]).dot(&qkv.slice(s![.., n_embd..n_embd*2]).t());

//     let n_embd = n_embd as f32;
//     let qk = qk.mapv(|x| x/n_embd.sqrt());

//     // TODO: masking 이후 softmax
// }

pub fn layer_norm(input: &Array2<f32>, weight: &Array1<f32>, bias: &Array1<f32>, config: &Config) -> Array2<f32> {
    let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let var = input.var_axis(Axis(1), 0.).insert_axis(Axis(1));
    let eps = config.layer_norm_epsilon;

    let output = (input-&mean) / (var+eps).mapv(|x| x.sqrt());

    (output * weight) + bias
}

pub fn softmax(qk: Array2<f32>) -> Array2<f32> {
    
    qk.fold_axis(Axis(1), f32::NEG_INFINITY, |&acc, &x| acc.max(x)); // 각 행의 가장 큰 값만 남기기
    qk.mapv(|x| x.exp())

}

// pub fn gelu() {

// }

