import dotenv
from rigl_torch.datasets import get_dataloaders
from hydra import initialize, compose

dotenv.load_dotenv("./.env")
with initialize("../configs", version_base="1.2.0"):
    cfg = compose("config.yaml", overrides=["dataset=imagenet"])
train_loader, test_loader = get_dataloaders(cfg)
print(f"length of train  laoder: {len(train_loader)}")
