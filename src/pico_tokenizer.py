from transformers import AutoTokenizer

# class PicoTokenizer:
#     def __init__(self):
#         pass


model_id = "allenai/OLMo-2-1124-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(tokenizer)