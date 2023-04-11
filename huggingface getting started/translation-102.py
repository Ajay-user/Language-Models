# https://huggingface.co/docs/transformers/main/model_doc/m2m_100
# https://huggingface.co/docs/transformers/multilingual#xlm-without-language-embeddings

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

en_text = "write in Hindi."
hindi_text = "हिंदी में लिखो "

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="hi")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")



encoded_hi = tokenizer(hindi_text, return_tensors="pt")


generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"),  max_new_tokens=50)
output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


print(output)


