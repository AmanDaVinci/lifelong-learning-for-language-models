import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score


class MbPA(nn.Module):
    
    def __init__(self,
                 base_model: nn.Module,
                 device: torch.device):
        super().__init__()
        self.memory = {}
        self.base_model = base_model
        self.dropout = nn.Dropout(self.base_model.config.hidden_dropout_prob)
        self.head = nn.Linear(self.base_model.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = device
        self.to(device)
      
    def add(self, batch, batch_keys, probability):
        if probability >= random.random(): 
            for idx, key in enumerate(batch_keys):
                self.memory[key.tobytes()] = (
                    batch["input_ids"][idx],
                    batch["attention_mask"][idx],
                    batch["token_type_ids"][idx],
                    batch["label"][idx],
                )

    def make_batches(self, input_ids, attention_mask, token_type_ids, labels, batch_size):
        batches = []
        last_idx = 0
        num_batches = int(len(input_ids)/batch_size)
        for i in range(1, num_batches+1):
            next_idx = (i*batch_size)-1
            batch = {
                "input_ids": torch.stack(input_ids[last_idx:next_idx]),
                "attention_mask": torch.stack(attention_mask[last_idx:next_idx]),
                "token_type_ids": torch.stack(token_type_ids[last_idx:next_idx]),
                "label": torch.stack(labels[last_idx:next_idx])
            }
            batches.append(batch)
            last_idx = next_idx
        return batches

    def sample(self, num_batches, batch_size):
        sample_size = num_batches * batch_size
        keys = random.sample(list(self.memory), sample_size)
        input_ids, attention_mask, token_type_ids, labels = [], [], [], []
        for key in keys:
            input_ids.append(self.memory[key][0])
            attention_mask.append(self.memory[key][1])
            token_type_ids.append(self.memory[key][2])
            labels.append(self.memory[key][3])
        return self.make_batches(input_ids, attention_mask, token_type_ids, labels, batch_size)
    
    def get_neighbours(self, batch, batch_keys):
        total_size = len(self.memory)
        all_keys = np.asarray(list(self.memory.keys()))
        all_keys = np.frombuffer(all_keys, dtype=np.float32).reshape(total_size, -1)
        key_neighbour_batches = []
        bs = len(batch["label"])
        for key in batch_keys:
            similarity_scores = np.dot(all_keys, key.T)
            neighbour_keys = all_keys[
                np.argpartition(similarity_scores, -bs)[-bs:]
            ]
            n_input_ids, n_attention_mask, n_token_type_ids, n_labels = [], [], [], []
            for nkey in neighbour_keys:
                nkey =  nkey.tobytes()
                n_input_ids.append(self.memory[nkey][0])
                n_attention_mask.append(self.memory[nkey][1])
                n_token_type_ids.append(self.memory[nkey][2])
                n_labels.append(self.memory[nkey][3])
            neighbour_batch = {
                "input_ids": torch.stack(n_input_ids),
                "attention_mask": torch.stack(n_attention_mask),
                "token_type_ids": torch.stack(n_token_type_ids),
                "label": torch.stack(n_labels)
            }
            key_neighbour_batches.append(neighbour_batch)
        return key_neighbour_batches
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids, 
                                  return_dict=True)
        pooler_output = outputs['pooler_output']
        logits = self.head(pooler_output)
        outputs['logits'] = logits
        return outputs

    def step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        labels = batch.pop("label")
        outputs = self.forward(**batch)
        logits = outputs['logits']
        loss = self.loss_fn(logits.float(), labels.unsqueeze(1).float())
        labels = labels.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        predictions = logits >= 0.0 # probability above 0.5
        accuracy = accuracy_score(labels, predictions)
        return loss, accuracy
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)