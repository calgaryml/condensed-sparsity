import torch
from torch.utils import benchmark
import pickle
from typing import List
from copy import deepcopy
import torch.nn as nn
from hydra import initialize, compose
from rigl_torch.models import ModelFactory
from rigl_torch.utils.checkpoint import Checkpoint
import dotenv
import os
import pathlib

from condensed_sparsity.v2.condensed_linear import (
    CondensedLinearStructured,
    CondensedLinearFineGrained,
    CondensedLinearFineGrainedSparseOp,
)

__MIN_RUN_TIME = 10


@torch.no_grad()
def main(
    mods: List[nn.Module],
    sparsities: List[float],
    device,
    cuda,
    num_features,
    dtype,
    num_threads,
):
    batch_sizes = [2**x for x in range(10, -1, -1)]
    results = []
    counter = 0
    for mod, sparsity in zip(mods, sparsities):
        mod.eval()
        for batch_size in batch_sizes:
            print(
                f"Benchmarking batch size {batch_size} with sparsity {sparsity}"
                f" and num_threads {num_threads}"
            )
            counter += 1
            sub_label = f"{batch_size:<6} x {num_features:<4}"
            x = benchmark.FuzzedTensor(
                "x",
                size=(batch_size, num_features),
                cuda=cuda,
                dtype=dtype,
                probability_contiguous=1.0,  # TRY contig.
            )
            x = x.default_tensor_constructor(x._size, x._dtype)
            x = x.to(device=device)
            cl_struc = CondensedLinearStructured(deepcopy(mod))
            cl_fine = CondensedLinearFineGrained(deepcopy(mod))
            cl_sparse_op = CondensedLinearFineGrainedSparseOp(deepcopy(mod))
            structured_cl = torch.jit.trace(  # TODO: debug starting here
                cl_struc.forward, x
            )  # TODO: Try script here instead
            fine_grained_cl = torch.jit.trace(cl_fine.forward, x)
            cl_sparse_op = torch.jit.trace(cl_sparse_op.forward, x)

            # Benchmarking begins...
            _ = structured_cl(x)  # Warmup
            results.append(
                benchmark.Timer(
                    stmt="structured_cl(x)",
                    setup="",
                    globals={"x": x, "structured_cl": structured_cl},
                    label="Condensed Linear",
                    sub_label=sub_label,
                    description=f"Structured sparsity @ {sparsity}",
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )
            _ = fine_grained_cl(x)
            results.append(
                benchmark.Timer(
                    stmt="fine_grained_cl(x)",
                    setup="",
                    globals={"x": x, "fine_grained_cl": fine_grained_cl},
                    label="Condensed Linear",
                    sub_label=sub_label,
                    description=(
                        f"Fine-grained + structured sparsity @ {sparsity}"
                    ),
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )
            _ = cl_sparse_op(x)
            results.append(
                benchmark.Timer(
                    stmt="cl_sparse_op(x)",
                    setup="",
                    globals={"x": x, "cl_sparse_op": cl_sparse_op},
                    label="Condensed Linear",
                    sub_label=sub_label,
                    description=(
                        f"structured sparsity + sparse op @ {sparsity}"
                    ),
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )
            _ = mod(x)
            results.append(
                benchmark.Timer(
                    stmt="mod(x)",
                    setup="",
                    globals={"x": x, "mod": mod},
                    label="Condensed Linear",
                    sub_label=sub_label,
                    description="Dense benchmark",
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )

    compare = benchmark.Compare(results)
    f_name = f"benchmark_v2_gpu_threads_{num_threads}.pkl"
    if device == "cpu":
        f_name = f"benchmark_v2_cpu_threads_{num_threads}.pkl"
    with open(f_name, "wb") as handle:
        pickle.dump(compare, handle)
    compare.colorize()
    print(compare)
    return results


def get_mod(run_id: str, device):
    with initialize("../configs", version_base="1.2.0"):
        cfg = compose(
            "config.yaml",
            overrides=[
                "compute.distributed=False",
                "dataset=imagenet",
                "model=vit",
                f"experiment.run_id={run_id}",
                "training.batch_size=2",
            ],
        )
    dotenv.load_dotenv("../.env", override=True)
    os.environ["IMAGE_NET_PATH"]
    checkpoint_dir = pathlib.Path(f"./artifacts/checkpoints/20230601_{run_id}")
    checkpoint = Checkpoint.load_best_checkpoint(checkpoint_dir=checkpoint_dir)
    model_state = checkpoint.model
    model = ModelFactory.load_model(
        model=cfg.model.name, dataset=cfg.dataset.name, diet=cfg.rigl.diet
    )
    model.to(device)
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        model_state = (
            checkpoint.get_single_process_model_state_from_distributed_state()
        )
        model.load_state_dict(model_state)
    return model.get_submodule("encoder.layers.encoder_layer_11.mlp.0")


if __name__ == "__main__":
    # for d in ["cuda:0", "cpu"]:
    __RUN_IDS = {90: "nrblbn15"}
    # __RUN_IDS = {90: "nrblbn15", 80: "0p0wrlb0"}
    # for num_threads in [1, 2, 4, 8, 16, 32]:
    for num_threads in [1]:
        # for d in ["cpu", "gpu"]:
        for d in ["cpu"]:
            if d == "cpu":
                device = torch.device("cpu")
                cuda = True
            else:
                device = torch.device("cuda")
                cuda = False
            mods, sparsities = [], []

            for sparsity, run_id in __RUN_IDS.items():
                mod = get_mod(run_id, device)
                mod.to(device)
                mods.append(mod)
                sparsities.append(sparsity)
                num_features = mod.weight.shape[1]
                dtype = torch.float32
                num_threads = num_threads
            results = main(
                mods, sparsities, device, cuda, num_features, dtype, num_threads
            )
