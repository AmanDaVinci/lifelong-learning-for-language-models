import random
import torch
import json
import numpy as np
from pathlib import Path
from functools import partial
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream, load_dataset_ids
from LL4LM.model import LifelongLearner
from LL4LM.trainers.trainer import Trainer

import wandb
import logging
log = logging.getLogger(__name__)


class MixtureOfExpertsTrainer(Trainer):

    def __init__(self, config: dict):
        super().__init__(config)

    def load_data(self):
        config = self.config.data
        self.dataset_ids, self.testset_ids = load_dataset_ids(
            multitask=config.multitask, 
            multilingual=config.multilingual, 
            multidomain=config.multidomain,
            shuffle=False # keep the ids ordered
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
        self.dataloaders = datastream.get_dataloader(
            self.tokenizer, 
            batch_size=config.batch_size,
            concatenate=False, # dataloader for each dataset
            shuffle_examples=True # shuffle each dataset
        )
        self.testloaders = teststream.get_dataloader(
            self.tokenizer, 
            batch_size=config.test_batch_size,
            concatenate=False,
            shuffle_examples=False
        )
        log.info(f"Loaded each dataset separately from Data Stream")

    def run(self):
        batch_size = self.config.data.batch_size
        test_every_nsteps = self.config.test_every_nsteps
        accumulate_every_nsteps = self.config.accumulate_gradient_every_nsteps
        format_dict = partial(json.dumps, indent=4)
        examples_seen = 0
        for dataset_id, dataloader, testloader in zip(self.dataset_ids, self.dataloaders, self.testloaders):
            self.load_model()
            self.model.train()
            log.info(f"Start training new model on {dataset_id}")
            wandb.watch(self.model, log="gradients", log_freq=test_every_nsteps)
            dataset_examples_seen = 0
            for batch in dataloader:
                examples_seen += batch_size
                dataset_examples_seen += batch_size
                loss, acc = self.model.step(batch)
                wandb.log(
                    {
                        f"train/{dataset_id}/loss": loss.item(),
                        f"train/{dataset_id}/accuracy": acc,
                        f"{dataset_id}_examples_seen": dataset_examples_seen,
                    }, 
                    step=examples_seen
                )
                loss = loss / accumulate_every_nsteps
                loss.backward()
                if i % accumulate_every_nsteps == 0:
                    self.opt.step()
                    self.model.zero_grad()
            self.model.zero_grad()
            save_path = self.ckpt_dir/f"{wandb.run.id}-{dataset_id}.pt"
            self.model.save(save_path)
            log.info(f"Trained model saved at {save_path}")
            loss, acc = self.test(dataset_id, testloader)
            wandb.log({f"test/{dataset_id}/loss": loss.item()}, step=examples_seen)
            wandb.log({f"test/{dataset_id}/accuracy": acc}, step=examples_seen)
            log.info(f"Test Accuracy on {dataset_id}: {acc}")
        log.info(f"Done training on all datasets.")

    def test(self, dataset_id, testloader):
        self.model.eval()
        losses, accuracies = [], [] 
        for batch in testloader:
            with torch.no_grad():
                loss, acc = self.model.step(batch)
            losses.append(loss.item())
            accuracies.append(acc)
        testset_loss = np.mean(losses)
        testset_accuracy = np.mean(accuracies)
        self.model.train()
        return testset_loss, testset_accuracy