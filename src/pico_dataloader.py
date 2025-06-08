import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer



class picoDataLoader:
    def __iter__(
        self, 
        dataset_id: str = 'mlfoundations/dclm-baseline-1.0',
        model_id: str = "allenai/OLMo-2-1124-7B-Instruct",
        max_len: int = 512,
        ):
        dataset = load_dataset(dataset_id, split='train', streaming=True)
        shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        keep_columns = ['metadata', 'text']
        shuffled_dataset = shuffled_dataset.remove_columns([col for col in shuffled_dataset.column_names if col not in keep_columns])
        self.dataset_iterator = iter(shuffled_dataset)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # print(self.tokenizer.build_inputs_with_special_tokens)
        self.max_len = max_len
        
        return self
    
    def __next__(self):
        next_tokens = next(self.dataset_iterator)['text']
        next_tokens = '<|endoftext|>' + next_tokens + '<|endoftext|>'
        inputs = self.tokenizer(
            next_tokens, 
            add_special_tokens=True, 
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            padding_side="left",
            )
        detokenized = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)
        print("".join(detokenized))
        return inputs
    