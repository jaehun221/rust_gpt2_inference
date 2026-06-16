use crate::config::Config;
use std::fs;

mod config;

fn main() {
    let config_path = fs::read_to_string("models/config.json").expect("models/config.json read err");
    let config: Config = Config::set_json(&config_path);

    println!("{:?}", config);
    
}
