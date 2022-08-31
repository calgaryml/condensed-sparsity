# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: 'Python 3.8.10 (''.venv'': venv)'
#     language: python
#     name: python3
# ---

import dotenv
from rigl_torch.datasets import get_dataloaders
from hydra import initialize, compose


dotenv.load_dotenv("../.env")
with initialize("../configs", version_base="1.2.0"):
    cfg = compose("config.yaml", overrides=["dataset=imagenet"])
train_loader, test_loader = get_dataloaders(cfg)
