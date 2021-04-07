import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from LL4LM.trainer import Trainer

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    with wandb.init(project="LL4LM", config=config):
        print(OmegaConf.to_yaml(config))
        trainer = Trainer(config)
        trainer.run()


if __name__ == '__main__':
    main()
