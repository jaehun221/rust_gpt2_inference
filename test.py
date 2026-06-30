from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer("Hello", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0]))