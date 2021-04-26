import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from LL4LM.trainers.lifelong_trainer import LifelongTrainer
from LL4LM.trainers.lifelong_trainer import ReplayTrainer
from LL4LM.trainers.multitask_trainer import MultitaskTrainer
from LL4LM.trainers.mixture_of_experts_trainer import MixtureOfExpertsTrainer
from LL4LM.trainers.datastream_scanner import DatastreamScanner


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    dict_config = OmegaConf.to_container(config, resolve=True)
    with wandb.init(project="LL4LM", config=dict_config):
        if config.trainer == "LifelongTrainer":
            trainer = LifelongTrainer(config)
        elif config.trainer == "ReplayTrainer":
            trainer = ReplayTrainer(config)
        elif config.trainer == "MultitaskTrainer":
            trainer = MultitaskTrainer(config)
        elif config.trainer == "MixtureOfExpertsTrainer":
            trainer = MixtureOfExpertsTrainer(config)
        elif config.trainer == "DatastreamScanner":
            trainer = DatastreamScanner(config)
        else:
            raise NotImplementedError(f"{config.trainer} not implemented.")
        trainer.run()


if __name__ == '__main__':
    main()
