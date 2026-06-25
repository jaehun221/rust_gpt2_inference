use crate::{config::Config, tokenizer::tokenizer, weights::Weights};
use std::fs;

mod config;
mod weights;
mod tokenizer;
mod operation;
mod structure;

fn main() {
    let config_path = fs::read_to_string("models/config.json").expect("models/config.json read err");
    let config: Config = Config::set_json(&config_path);

    println!("{:?}", config);

    let weights = Weights::weights_load("models/model.safetensors");
    // weights.tensor_list();
    println!("{:?}", weights.get_tensor1("h.7.mlp.c_proj.bias").shape());
}
