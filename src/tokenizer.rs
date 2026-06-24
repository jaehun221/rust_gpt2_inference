use tokenizers::tokenizer::Tokenizer;


pub fn tokenizer() {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").expect("tokenizer.json not found");
    let output = tokenizer.encode("Hello, y'all! How are you?", true).expect("encode err");

    println!("{:?}", output.get_ids());

}