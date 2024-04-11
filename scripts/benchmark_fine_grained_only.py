import torch
from torch.utils import benchmark
import pickle
from copy import deepcopy
from hydra import initialize, compose
from rigl_torch.models import ModelFactory
from rigl_torch.utils.checkpoint import Checkpoint
import dotenv

# import os
import pathlib

from condensed_sparsity.condensed_linear import (  # noqa
    CSRLinear,
    CondensedLinearStructured,
    CondensedLinearFineGrained,
    VmapCondensed,
    FixedFanInCuda,
    CondensedLinearFineGrainedSparseOp,  # noqa
)

__MIN_RUN_TIME = 2


def get_mod(
    run_id: str,
    device,
    layer_name: str = "encoder.layers.encoder_layer_11.mlp.3",
):
    with initialize("../configs", version_base="1.2.0"):
        cfg = compose(
            "config.yaml",
            overrides=[
                "compute.distributed=False",
                "dataset=imagenet",
                "model=vit",
                f"experiment.run_id={run_id.split('_')[1]}",
                "training.batch_size=2",
            ],
        )
    dotenv.load_dotenv("../.env", override=True)
    checkpoint_dir = pathlib.Path(f"./artifacts/checkpoints/{run_id}")
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
    return model.get_submodule(layer_name)


if __name__ == "__main__":
    __RUN = {
        90: "20230601_nrblbn15",
    }
    __LAYER_NAME = "encoder.layers.encoder_layer_11.mlp.3"
    # __DEVICE = "CPU"
    __DEVICE = "GPU"
    __FNAME = "benchmark_test.pkl"
    __NUM_THREADS = 4
    if __DEVICE.lower() == "cpu":
        device = torch.device("cpu")
        cuda = False
    else:
        device = torch.device("cuda")
        cuda = True

    torch.jit.enable_onednn_fusion(
        True
    )  # seems like we need this to use inductor on gpu

    mod = get_mod(__RUN[90], device, __LAYER_NAME)
    batch_size = 1
    dtype = torch.float32
    num_features = mod.weight.shape[1]
    x = benchmark.FuzzedTensor(
        "x",
        size=(batch_size, num_features),
        cuda=cuda,
        dtype=dtype,
        probability_contiguous=1.0,
    )
    x = x.default_tensor_constructor(x._size, x._dtype)
    x = x.to(device=device)
    results = []

    cl_fine = CondensedLinearFineGrained(deepcopy(mod), dtype=dtype).eval()
    compiler_kwargs = {
        "mode": "max-autotune",
        "fullgraph": True,
    }
    # dense_compiled = torch.compile(mod, backend="inductor", **compiler_kwargs)
    cl_fine_compiled = torch.compile(
        cl_fine, backend="inductor", **compiler_kwargs
    )
    # for _ in range(10):
    _ = cl_fine_compiled(x)  # jit warmup

    results.append(
        benchmark.Timer(
            stmt="cl_fine_compiled(x)",
            setup="",
            globals={
                "x": x,
                "cl_fine_compiled": cl_fine_compiled,
            },
            label=f"Benchmark for {x.shape}",
            sub_label="Condensed fine-grained linear",
            description=("Fine-grained + structured with backend inductor"),
            num_threads=__NUM_THREADS,
        ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
    )
    # for _ in range(10):
    # _ = dense_compiled(x)  # jit warmup
    # results.append(
    #     benchmark.Timer(
    #         stmt="dense_compiled(x)",
    #         setup="",
    #         globals={
    #             "x": x,
    #             "dense_compiled": dense_compiled,
    #         },
    #         label=f"Benchmark for {x.shape}",
    #         sub_label="Dense",
    #         description=("Dense with backend inductor"),
    #         num_threads=__NUM_THREADS,
    #     ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
    # )

    compare = benchmark.Compare(results)
    with open(__FNAME, "wb") as handle:
        pickle.dump(compare, handle)
    compare.colorize()
    print(compare)
