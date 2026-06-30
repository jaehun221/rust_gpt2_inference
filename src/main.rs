use crate::{config::Config, structure::KVCache, tokenizer::encode, weights::Weights};
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
    let mut caches = KVCache::cache_set(&config);
    let mut next = gpt2.generate(&token, &config, &mut caches);

    for _ in 0..max_token {
        if next == 50256 {
            break;
        }

        token.push(next);

        next = gpt2.generate(&vec![next], &config, &mut caches);
    }

    let text = tokenizer::decode(&token);
    println!("{}", text);
}
