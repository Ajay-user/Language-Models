# https://huggingface.co/docs/transformers/main/model_doc/m2m_100
# https://huggingface.co/docs/transformers/multilingual#xlm-without-language-embeddings

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")



encoded_zh = tokenizer(chinese_text, return_tensors="pt")


generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"), max_new_tokens=50)
output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


print(output)


