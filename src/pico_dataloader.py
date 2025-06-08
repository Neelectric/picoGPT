import torch
from datasets import load_dataset
from tqdm import tqdm



dataset = load_dataset('mlfoundations/dclm-baseline-1.0', split='train', streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
keep_columns = ['metadata', 'text']
shuffled_dataset = shuffled_dataset.remove_columns([col for col in shuffled_dataset.column_names if col not in keep_columns])


dataset_iterator = iter(shuffled_dataset)
liste = []
for i in tqdm(range(5000), dynamic_ncols=True):
    liste.append(next(dataset_iterator)['text'])