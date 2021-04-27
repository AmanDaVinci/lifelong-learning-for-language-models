import json
import numpy as np
from functools import partial

from LL4LM.models.replay_memory import ReplayMemory
from LL4LM.trainers.lifelong_trainer import LifelongTrainer

import wandb
import logging
log = logging.getLogger(__name__)


class ReplayTrainer(LifelongTrainer):
    
    def run(self):
        replay_memory = ReplayMemory(first_batch=next(iter(self.dataloader)))
        self.model.train()
        self.model.zero_grad()
        batch_size = self.config.data.batch_size
        test_every_nsteps = self.config.test_every_nsteps
        replay_every_nsteps = self.config.trainer.replay_every_nsteps
        num_replay_batches = self.config.trainer.num_replay_batches
        add_probability = self.config.trainer.replay_add_probability
        # log metrics before training starts
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
            replay_memory.add(batch, add_probability)
            if (i+1) % replay_every_nsteps == 0:
                for batch in replay_memory.sample(num_replay_batches):
                    loss, acc = self.model.step(batch)
                    loss.backward()
                    self.opt.step()
                    self.model.zero_grad()
                    wandb.log({
                        "replay/loss": loss.item(),
                        "replay/accuracy": acc,
                        "replay/memory_size": len(replay_memory)
                    })
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
        np.save(self.output_dir/"examples_seen.npy", np.array(examples_seen))
        save_path = self.ckpt_dir/f"{wandb.run.id}.pt"
        self.model.save(save_path)
        log.info(f"Trained model saved at {save_path}")
