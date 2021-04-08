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
        batch_size, test_freq = self.config.data.batch_size, self.config.test_freq
        examples_seen = 0
        for dataset_id, dataloader, testloader in zip(self.dataset_ids, self.dataloaders, self.testloaders):
            self.model.train()
            log.info(f"Start training model on {dataset_id}")
            wandb.watch(self.model, log="gradients", log_freq=test_freq)
            dataset_examples_seen = 0
            for batch in dataloader:
                loss, acc = self.model.step(batch)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                examples_seen += batch_size
                dataset_examples_seen += batch_size
                wandb.log(
                    {
                        f"train/{dataset_id}/loss": loss.item(),
                        f"train/{dataset_id}/accuracy": acc,
                        f"{dataset_id}_examples_seen": dataset_examples_seen,
                    }, 
                    step=examples_seen
                )
            save_path = self.ckpt_dir/f"{wandb.run.id}-{dataset_id}.pt"
            self.model.save(save_path)
            log.info(f"Trained model saved at {save_path}")
            loss, acc = self.test(dataset_id, testloader)
            wandb.log({f"test/{dataset_id}/loss": loss.item()}, step=examples_seen)
            wandb.log({f"test/{dataset_id}/accuracy": acc}, step=examples_seen)
            log.info(f"Test Accuracy on {dataset_id}: {acc}")
            self.load_model()
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