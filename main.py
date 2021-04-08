import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from LL4LM.trainer import Trainer

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    dict_config = OmegaConf.to_container(config, resolve=True)
    with wandb.init(project="LL4LM", config=dict_config):
        trainer = Trainer(config)
        trainer.run()


if __name__ == '__main__':
    main()
