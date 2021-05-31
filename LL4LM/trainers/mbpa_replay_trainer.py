import gc
import json
import time
import torch
import random
import numpy as np
from pathlib import Path
from copy import deepcopy
from torch.optim import SGD

from LL4LM.models.mbpa import MbPA
from LL4LM.datastreams import DataStream
from transformers import AdamW, AutoTokenizer, AutoModel

import wandb
import logging
log = logging.getLogger(__name__)


class MbPAReplayTrainer():

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.output_dir)/wandb.run.id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.set_seed(config.seed)
        self.load_data()
        self.load_model()

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"Setting seed: {seed}")

    def load_data(self):
        config = self.config.data
        self.dataset_names = self.config.datastream
        datastream = DataStream(self.dataset_names, split="train_split")
        teststream = DataStream(self.dataset_names, split="test_split")
        if config.shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.resize_datasets(config.dataset_size)
        teststream.limit_datasets(config.testset_size)
        examples = datastream.sample_examples(config.n_samples_each_dataset)
        wandb.log({"Sampled_Examples": wandb.Table(dataframe=examples)}, step=0)
        wandb.log({"Data_Stream": wandb.Table(dataframe=datastream.summary())}, step=0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.dataloader = datastream.get_dataloader(
            self.tokenizer, 
            batch_size=config.batch_size,
            concatenate=True,
            shuffle_examples=False
        )
        self.testloaders = teststream.get_dataloader(
            self.tokenizer, 
            batch_size=config.test_batch_size,
            concatenate=False,
            shuffle_examples=False
        )
        self.dataloader_keys = get_dataloader_keys(self.dataloader, self.config.base_model)
        self.testloaders_keys = [get_dataloader_keys(dl, self.config.base_model) for dl in self.testloaders]
        log.info(f"Loaded Data Stream")

    def load_model(self): 
        config = self.config.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = AutoModel.from_pretrained(config.base_model)
        self.model = MbPA(base_model, device)
        log.info(f"Loaded {config.base_model} on {device}")
        self.opt = AdamW(
            self.model.parameters(), 
            weight_decay=config.wt_decay,
            lr=config.lr
        )

    def run(self):
        # making sure memory is not empty
        self.model.add(next(iter(self.dataloader)), self.dataloader_keys[0], probability=1.0)
        self.model.train()
        self.model.zero_grad()
        batch_size = self.config.data.batch_size
        test_interval = self.config.test_interval
        replay_interval = self.config.trainer.replay_interval
        num_replay_batches = self.config.trainer.num_replay_batches
        add_probability = self.config.trainer.replay_add_probability
        def _test_log():
            start = time.perf_counter()
            losses, accuracies = self.test()
            log.info(f"Testing done in {time.perf_counter()-start:.04f} secs")
            wandb.log(losses, step=examples_seen)
            wandb.log(accuracies, step=examples_seen)
            index.append(examples_seen)
            head_weights.append(self.model.head.weight.detach().cpu().numpy())
            head_biases.append(self.model.head.bias.detach().cpu().numpy())
            log.info(
                f"Test Accuracies at {examples_seen}:"\
                f"{json.dumps(accuracies, indent=4)}"
            )
        _test_log()
        examples_seen = 0
        for i, batch in enumerate(self.dataloader):
            examples_seen += batch_size
            self.model.zero_grad()
            loss, acc = self.model.step(batch)
            loss.backward()
            self.opt.step()
            wandb.log({"train/loss": loss.item()}, step=examples_seen)
            wandb.log({"train/accuracy": acc}, step=examples_seen)
            model.add(batch, self.dataloader_keys[i], add_probability)
            if (i+1) % replay_interval == 0:
                for batch in model.sample(num_replay_batches, batch_size):
                    loss, acc = self.model.step(batch)
                    loss.backward()
                    self.opt.step()
                    self.model.zero_grad()
            if (i+1) % test_interval == 0:
                _test_log()
        self.model.zero_grad()
        _test_log()

    def test(self):
        model_path = self.output_dir/"model.ckpt"
        self.model.save(model_path)
        self.model.to("cpu")
        del(self.model)
        del(self.opt)
        gc.collect()
        torch.cuda.empty_cache()
        self.model.eval()
        testset_losses, testset_accuracies  = {}, {}
        for name, dl, keys in zip(self.dataset_names, self.testloaders, self.testloaders_keys):
            losses, accuracies = [], [] 
            for idx, batch in enumerate(dl):
                neighbour_batches = model.get_neighbours(batch, keys[idx])
                input_ids, attention_masks, token_type_ids, labels = batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["label"]
                for i in range(self.config.data.batch_size):
                    unit_batch = {
                        "input_ids": torch.unsqueeze(input_ids[i],0),
                        "attention_mask": torch.unsqueeze(attention_masks[i],0),
                        "token_type_ids": torch.unsqueeze(token_type_ids[i],0),
                        "label": torch.unsqueeze(labels[i],0)
                    }
                    neighbour_batch = neighbour_batches[i]
                    loss, acc = local_adaptation(unit_batch, neighbour_batch, model_path, self.config)
                    losses.append(loss.item())
                    accuracies.append(acc)
            testset_losses[f"test/{name}/loss"] = np.mean(losses)
            testset_accuracies[f"test/{name}/accuracy"] = np.mean(accuracies)
        self.load_model()
        self.model.load(model_path)
        self.model.train()
        return testset_losses, testset_accuracies
    
def local_adaptation(batch, neighbour_batch, model_path, config):
    base_model = AutoModel.from_pretrained(config.model.base_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp_model = MbPA(base_model, device)
    temp_model.load(model_path)
    with torch.no_grad():
        orig_params = torch.cat([torch.reshape(param, [-1]) for param in temp_model.parameters()], axis=0).to("cpu")
    opt = SGD(
        temp_model.parameters(), 
        lr=config.model.lr
    )
    temp_model.train()
    temp_model.zero_grad()
    for step in range(config.trainer.adapt_steps):
        with torch.no_grad():
            params = torch.cat([torch.reshape(param, [-1]) for param in temp_model.parameters()], axis=0).to("cpu")
        temp_model.zero_grad()
        loss, _ = temp_model.step(neighbour_batch)
        loss += config.trainer.adapt_lambda * torch.sum((orig_params - params)**2)
        loss.backward()
        opt.step()
    temp_model.eval()
    with torch.no_grad():
        test_loss, test_accuracy = temp_model.step(batch)
    temp_model.to("cpu")
    del(loss)
    del(opt)
    del(temp_model)
    gc.collect()
    torch.cuda.empty_cache()
    return test_loss, test_accuracy

def get_dataloader_keys(dataloader, encoder_name):
    dataloader_keys = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    key_encoder = AutoModel.from_pretrained(encoder_name).to(device)
    key_encoder.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = key_encoder(
                batch["input_ids"],
                batch["attention_mask"], 
                batch["token_type_ids"], 
                return_dict=True
            )
            batch_cls_features = outputs['pooler_output'].detach().cpu().numpy()
        dataloader_keys.append(batch_cls_features)
    key_encoder.to("cpu")
    del(key_encoder)
    gc.collect()
    torch.cuda.empty_cache()
    return dataloader_keys