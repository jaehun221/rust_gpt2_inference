use tokenizers::tokenizer::Tokenizer;


pub fn encode(input: &str) -> Vec<usize> {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").expect("tokenizer.json not found");
    let output = tokenizer.encode(input, true).expect("encode err");

    let token_id = output.get_ids().iter().map(|&x| x as usize).collect();

    token_id
}

pub fn decode(ids: &Vec<usize>) -> String {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").expect("tokenizer.json not found");
    let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
    tokenizer.decode(&ids_u32, true).expect("decode err")
}