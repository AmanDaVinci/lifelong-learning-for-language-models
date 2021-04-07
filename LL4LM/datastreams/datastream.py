import random
import pandas as pd
from functools import partial
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from LL4LM.datastreams.loaders import load_datastream


class DataStream:

    def __init__(self, dataset_ids: list):
        self.ids = dataset_ids
        self.stream = [
            load_datastream(id.path, id.name, id.split, id.filter_column, id.filter_value) 
            for id in dataset_ids
        ]
    
    # TODO: better name
    def df(self):
        return pd.DataFrame(
            [(str(id), data.num_rows) for id, data in zip(self.ids, self.stream)],
            columns=["dataset", "num_examples"]
        )

    def sample_examples(self, num_per_dataset: int=1) -> pd.DataFrame:
        all_sample_data = []
        for id, data in zip(self.ids, self.stream):
            sample_idxs = random.choices(range(data.num_rows), k=num_per_dataset)
            sample_data = data.select(sample_idxs).to_pandas()
            sample_data["dataset"] = str(id)
            all_sample_data.append(sample_data)
        return pd.concat(all_sample_data)

    def shuffle_datasets(self, seed: int=None):
        self.stream = [data.shuffle(seed) for data in self.stream]
    
    def limit_datasets(self, max_size: int):
        self.stream = [data.select(range(max_size)) if max_size<=data.num_rows else data
                       for data in self.stream]
    
    def remix_datasets(self, indices: list):
        assert len(self.stream) == len(indices), \
            "Must have indices for each dataset in the datastream."
        self.stream = [data.select(idxs) if max(idxs)<=data.num_rows else data
                       for data, idxs in zip(self.stream, indices)]
    
    def get_dataloader(self, tokenizer, concatenate: bool, batch_size: int, shuffle_examples: bool):
        tokenizer = partial(tokenizer.batch_encode_plus, 
                            padding="max_length", truncation="only_first")
        def dataloader(dataset):
            dataset = dataset.map(lambda x: tokenizer(list(zip(x["context"], x["statement"]))),
                                  batched=True, remove_columns=["context", "statement"])
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 
                                                      'attention_mask', 'label'])
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_examples)
        if concatenate:
            # BUG: https://github.com/huggingface/datasets/pull/2025
            # HOTFIX: Create and cache a new dataset using flatten_indices()
            self.stream = [data.flatten_indices() for data in self.stream]
            return dataloader(concatenate_datasets(self.stream))
        else:
            return [dataloader(dataset) for dataset in self.stream]
