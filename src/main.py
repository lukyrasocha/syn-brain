# Test wandb with Hydra integration
import random
import wandb
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(
        project=cfg.wandb.project,
        config={
            "learning_rate": cfg.hyperparams.learning_rate,
            "architecture": cfg.model.architecture,
            "dataset": cfg.data.dataset,
            "epochs": cfg.training.epochs,
        },
    )

    run.finish()

if __name__ == "__main__":
    main()
