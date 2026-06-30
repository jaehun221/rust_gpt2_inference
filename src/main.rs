use crate::{config::Config, tokenizer::encode, weights::Weights};
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


    let mut token = encode("Hello");
    let max_token= 20;

    let gpt2 = structure::Gpt2::from_weight(&weights, &config);

    for i in 0..max_token {
        let next = gpt2.generate(&token, &config);

        if next == 50256 {
            break;
        }

        token.push(next);
    }

    let text = tokenizer::decode(&token);
    println!("{}", text);
}
