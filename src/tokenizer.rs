use tokenizers::tokenizer::Tokenizer;


pub fn tokenizer(input: &str) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").expect("tokenizer.json not found");
    let output = tokenizer.encode(input, true).expect("encode err");

    let token_id = output.get_ids().to_vec();

    token_id
}