use serde::{ Deserialize };

#[derive(Deserialize, Debug)]
pub struct Config {
    pub layer_norm_epsilon: f64,
    pub n_embd: usize,
    pub n_head: usize,
}

impl Config {
    pub fn set_json(path: &str) -> Self {
        serde_json::from_str(&path).expect("config.json Deserialize Err")
    }
}
