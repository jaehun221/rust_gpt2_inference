use tokenizers::tokenizer::Tokenizer;


pub fn tokenizer(input: &str) -> Vec<usize> {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").expect("tokenizer.json not found");
    let output = tokenizer.encode(input, true).expect("encode err");

    let token_id = output.get_ids().iter().map(|&x| x as usize).collect();

    token_id
}