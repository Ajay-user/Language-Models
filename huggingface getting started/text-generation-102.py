from transformers import GPT2Tokenizer, TFAutoModelForCausalLM


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")

tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer(["Today is"], return_tensors="tf")


outputs = model.generate(**inputs, max_new_tokens=10, return_dict_in_generate=True)


print(outputs)

print([tokenizer.decode(i) for i in outputs[0]])