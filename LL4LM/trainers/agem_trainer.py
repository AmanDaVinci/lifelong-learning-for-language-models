import time
import json
import torch
import numpy as np
from functools import partial

from LL4LM.models.replay_memory import ReplayMemory
from LL4LM.trainers.lifelong_trainer import LifelongTrainer
from LL4LM.utils.gradients import (
    project,
    pairwise_gradient_similarity, 
    sequential_gradient_interference
)

import wandb
import logging
log = logging.getLogger(__name__)


class AGEMTrainer(LifelongTrainer):
    
    def run(self):
        replay_memory = ReplayMemory(first_batch=next(iter(self.dataloader)))
        self.model.train()
        self.model.zero_grad()
        batch_size = self.config.data.batch_size
        test_interval = self.config.test_interval
        test_grad_interval = self.config.test_grad_interval
        train_grad_interval = self.config.train_grad_interval
        replay_interval = self.config.trainer.replay_interval
        num_replay_batches = self.config.trainer.num_replay_batches
        add_probability = self.config.trainer.replay_add_probability
        examples_seen = 0
        projected_grads, nonzero_indices = None, None
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
        def _test_grad_log():
            start = time.perf_counter()
            grad_sim, grad_shared = pairwise_gradient_similarity(self.model, self.dataset_names, self.gradloaders)
            log.info(f"Grad sim measured in {time.perf_counter()-start:.04f} secs")
            wandb.log(grad_sim, step=examples_seen)
            wandb.log(grad_shared, step=examples_seen)
        _test_log()
        _test_grad_log()
        for i, batch in enumerate(self.dataloader):
            examples_seen += batch_size
            self.model.zero_grad()
            loss, acc = self.model.step(batch)
            params = [p for p in self.model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            if (i+1) % replay_interval == 0:
                sum_replay_grads = [torch.zeros_like(g) for g in grads]
                for batch in replay_memory.sample(num_replay_batches):
                    loss, acc = self.model.step(batch)
                    replay_grads = torch.autograd.grad(loss, params)
                    sum_replay_grads = [x+y for x, y in zip(replay_grads, sum_replay_grads)]
                projected_grads = project(grads, sum_replay_grads)
            else:
                projected_grads = grads
            for param, grad in zip(params, projected_grads):
                param.grad = grad.data
            self.opt.step()
            replay_memory.add(batch, add_probability)
            wandb.log({"train/loss": loss.item()}, step=examples_seen)
            wandb.log({"train/accuracy": acc}, step=examples_seen)
            if (i+1) % train_grad_interval == 0:
                outputs = sequential_gradient_interference(self.model, projected_grads, nonzero_indices)
                projected_grads, nonzero_indices, interference, overlap = outputs
                wandb.log({"gradient/interference": interference}, step=examples_seen)
                wandb.log({"gradient/overlap": overlap}, step=examples_seen)
            if (i+1) % test_interval == 0:
                _test_log()
            if (i+1) % test_grad_interval == 0:
                _test_grad_log()
        self.model.zero_grad()
        _test_log()
        _test_grad_log()
        np.save(self.output_dir/"head_weights.npy", np.concatenate(head_weights))
        np.save(self.output_dir/"head_biases.npy", np.concatenate(head_biases))
        np.save(self.output_dir/"index.npy", np.array(index))
        save_path = self.ckpt_dir/f"{wandb.run.id}.pt"
        self.model.save(save_path)
        log.info(f"Trained model saved at {save_path}")
