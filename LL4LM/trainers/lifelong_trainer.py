import random
import torch
import numpy as np
from pathlib import Path
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream, load_dataset_ids
from LL4LM.model import LifelongLearner
from LL4LM.trainers.trainer import Trainer

import wandb
import logging
log = logging.getLogger(__name__)


class LifelongTrainer(Trainer):

    def __init__(self, config: dict):
        super().__init__(config)
    
    def load_data(self):
        config = self.config.data
        self.dataset_ids, self.testset_ids = load_dataset_ids(
            multitask=config.multitask, 
            multilingual=config.multilingual, 
            multidomain=config.multidomain,
            shuffle=config.stream_shuffle
        )
        datastream = DataStream(self.dataset_ids)
        teststream = DataStream(self.testset_ids)
        if config.data_shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.limit_datasets(config.max_size_each_dataset)
        teststream.limit_datasets(config.testset_size)
        examples = datastream.sample_examples(config.n_samples_each_dataset)
        wandb.log({"Sampled_Examples": wandb.Table(dataframe=examples)}, step=0)
        wandb.log({"Data_Stream": wandb.Table(dataframe=datastream.df())}, step=0)
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
        log.info(f"Loaded Data Stream")

    def run(self):
        self.model.train()
        log.info(f"Start training model")
        batch_size, test_freq = self.config.data.batch_size, self.config.test_freq
        wandb.watch(self.model, log="gradients", log_freq=test_freq)
        examples_seen = 0
        for i, batch in enumerate(self.dataloader):
            if i%test_freq==0:
                losses, accuracies = self.test()
                wandb.log(losses, step=examples_seen)
                wandb.log(accuracies, step=examples_seen)
                log.info(f"Test Accuracies after {examples_seen}: {accuracies}")
            loss, acc = self.model.step(batch)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            examples_seen += batch_size
            wandb.log({"train/loss": loss.item()}, step=examples_seen)
            wandb.log({"train/accuracy": acc}, step=examples_seen)
        save_path = self.ckpt_dir/f"{wandb.run.id}.pt"
        self.model.save(save_path)
        log.info(f"Trained model saved at {save_path}")

    def test(self):
        self.model.eval()
        testset_losses, testset_accuracies  = {}, {}
        for testset_id, testloader in zip(self.testset_ids, self.testloaders):
            losses, accuracies = [], [] 
            for batch in testloader:
                with torch.no_grad():
                    loss, acc = self.model.step(batch)
                losses.append(loss.item())
                accuracies.append(acc)
            testset_losses[f"test/{testset_id}/loss"] = np.mean(losses)
            testset_accuracies[f"test/{testset_id}/accuracy"] = np.mean(accuracies)
        self.model.train()
        return testset_losses, testset_accuracies
