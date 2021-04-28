import time
import json
import torch
import random
import numpy as np
from pathlib import Path
from functools import partial
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream
from LL4LM.models.lifelong_learner import LifelongLearner
from LL4LM.trainers.trainer import Trainer
from LL4LM.utils.gradients import gradient_similarity

import wandb
import logging
log = logging.getLogger(__name__)


class LifelongTrainer(Trainer):

    def __init__(self, config: dict):
        super().__init__(config)
    
    def load_data(self):
        config = self.config.data
        self.dataset_names = self.config.datastream
        datastream = DataStream(self.dataset_names, split="train_split")
        teststream = DataStream(self.dataset_names, split="test_split")
        if config.shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.limit_datasets(config.dataset_size)
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
        log.info(f"Loaded Data Stream")

    def run(self):
        self.model.train()
        self.model.zero_grad()
        batch_size = self.config.data.batch_size
        test_every_nsteps = self.config.test_every_nsteps
        gradsim_interval = self.config.gradsim_interval
        examples_seen = 0
        index, head_weights, head_biases = [], [], []
        losses, accuracies = self.test()
        wandb.log(losses, step=examples_seen)
        wandb.log(accuracies, step=examples_seen)
        index.append(examples_seen)
        head_weights.append(self.model.head.weight.detach().cpu().numpy())
        head_biases.append(self.model.head.bias.detach().cpu().numpy())
        format_dict = partial(json.dumps, indent=4)
        log.info(
            f"Test Accuracies before training:"\
            f"{format_dict(accuracies)}"
        )
        wandb.watch(self.model, log="gradients", log_freq=test_every_nsteps)
        for i, batch in enumerate(self.dataloader):
            examples_seen += batch_size
            loss, acc = self.model.step(batch)
            wandb.log({"train/loss": loss.item()}, step=examples_seen)
            wandb.log({"train/accuracy": acc}, step=examples_seen)
            loss.backward()
            self.opt.step()
            self.model.zero_grad()
            if (i+1) % test_every_nsteps == 0:
                losses, accuracies = self.test()
                wandb.log(losses, step=examples_seen)
                wandb.log(accuracies, step=examples_seen)
                index.append(examples_seen)
                head_weights.append(self.model.head.weight.detach().cpu().numpy())
                head_biases.append(self.model.head.bias.detach().cpu().numpy())
                log.info(
                    f"Test Accuracies after seeing {examples_seen} examples:"\
                    f"{format_dict(accuracies)}"
                )
            if (i+1) % gradsim_interval == 0:
                start = time.perf_counter()
                grad_sim, grad_shared = gradient_similarity(self.model, self.dataset_names, self.testloaders)
                log.info(f"Grad sim measured in {time.perf_counter()-start:.04f} secs")
                wandb.log(task_sim, step=examples_seen)
                wandb.log(task_shared, step=examples_seen)
        self.model.zero_grad()
        losses, accuracies = self.test()
        wandb.log(losses, step=examples_seen)
        wandb.log(accuracies, step=examples_seen)
        index.append(examples_seen)
        head_weights.append(self.model.head.weight.detach().cpu().numpy())
        head_biases.append(self.model.head.bias.detach().cpu().numpy())
        log.info(
            f"Final Test Accuracies after seeing {examples_seen} examples:"\
            f"{format_dict(accuracies)}"
        )
        np.save(self.output_dir/"head_weights.npy", np.concatenate(head_weights))
        np.save(self.output_dir/"head_biases.npy", np.concatenate(head_biases))
        np.save(self.output_dir/"index.npy", np.array(index))
        save_path = self.ckpt_dir/f"{wandb.run.id}.pt"
        self.model.save(save_path)
        log.info(f"Trained model saved at {save_path}")

    def test(self):
        self.model.eval()
        testset_losses, testset_accuracies  = {}, {}
        for name, testloader in zip(self.dataset_names, self.testloaders):
            losses, accuracies = [], [] 
            for batch in testloader:
                with torch.no_grad():
                    loss, acc = self.model.step(batch)
                losses.append(loss.item())
                accuracies.append(acc)
            testset_losses[f"test/{name}/loss"] = np.mean(losses)
            testset_accuracies[f"test/{name}/accuracy"] = np.mean(accuracies)
        self.model.train()
        return testset_losses, testset_accuracies
