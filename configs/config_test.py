import hydra
from omegaconf import DictConfig

@hydra.main(config_path = ".", config_name = "default", version_base="1.3")
def main(cfg: DictConfig):
    print("Hydra\n",cfg)
    
if __name__ == "__main__":
    main()