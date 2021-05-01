import time
import json
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream
from LL4LM.models.lifelong_learner import LifelongLearner
from LL4LM.trainers.trainer import Trainer
from LL4LM.utils.gradients import gradient_similarity, gradient_interference

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
        gradstream = DataStream(self.dataset_names, split="test_split")
        if config.shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.limit_datasets(config.dataset_size)
        teststream.limit_datasets(config.testset_size)
        gradstream.limit_datasets(config.gradset_size)
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
        self.gradloaders = gradstream.get_dataloader(
            self.tokenizer, 
            batch_size=config.grad_batch_size,
            concatenate=False,
            shuffle_examples=False
        )
        log.info(f"Loaded Data Stream")

    def run(self):
        self.model.train()
        self.model.zero_grad()
        batch_size = self.config.data.batch_size
        test_interval = self.config.test_interval
        gradsim_interval = self.config.gradsim_interval
        examples_seen = 0
        prev_grads = None
        index, head_weights, head_biases = [], [], []
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
        def _interference_log():
            interference, grads = gradient_interference(self.model, prev_grads)
            prev_grads = grads
            wandb.log(interference, step=examples_seen)
        def _gradsim_log():
            start = time.perf_counter()
            grad_sim, grad_shared = gradient_similarity(self.model, self.dataset_names, self.gradloaders)
            log.info(f"Grad sim measured in {time.perf_counter()-start:.04f} secs")
            wandb.log(grad_sim, step=examples_seen)
            wandb.log(grad_shared, step=examples_seen)
        _test_log()
        _gradsim_log()
        wandb.watch(self.model, log="gradients", log_freq=test_interval)
        for i, batch in enumerate(self.dataloader):
            examples_seen += batch_size
            loss, acc = self.model.step(batch)
            wandb.log({"train/loss": loss.item()}, step=examples_seen)
            wandb.log({"train/accuracy": acc}, step=examples_seen)
            loss.backward()
            self.opt.step()
            self.model.zero_grad()
            if (i+1) % test_interval == 0:
                _test_log()
            if (i+1) % self.config.interference_measurement_interval == 0:
                _interference_log()
            if (i+1) % gradsim_interval == 0:
                _gradsim_log()
        self.model.zero_grad()
        _test_log()
        _interference_log()
        _gradsim_log()
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
