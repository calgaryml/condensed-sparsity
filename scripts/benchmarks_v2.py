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

from condensed_sparsity.v2.condensed_linear import CondensedLinear

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
    batch_sizes = [2**x for x in range(8, -1, -1)]
    results = []
    counter = 0
    for mod, sparsity in zip(mods, sparsities):
        for batch_size in batch_sizes:
            print(
                f"Benchmarking batch size {batch_size} with sparsity {sparsity}"
            )
            counter += 1
            sub_label = f"{batch_size:<6} x {num_features:<4}"
            x = benchmark.FuzzedTensor(
                "x",
                size=(batch_size, num_features),
                cuda=cuda,
                dtype=dtype,
            )
            x = x.default_tensor_constructor(x._size, x._dtype)
            x = x.to(device=device)
            cl = CondensedLinear(deepcopy(mod))
            structured_cl = torch.jit.trace_module(  # TODO: debug starting here
                cl, x
            )  # TODO: Try script here instead
            fine_grained_cl = torch.jit.trace_module(cl.fine_grained_forward, x)

            # Benchmarking begins...
            results.append(
                benchmark.Timer(
                    stmt="structured_cl(x)",
                    setup=None,
                    globals={"x": x, "structured_cl": structured_cl},
                    label="Structured sparsity",
                    sub_label=sub_label,
                    description=f"Structured sparsity @ {sparsity}",
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )
            results.append(
                benchmark.Timer(
                    stmt="fine_grained_cl.fine_grained_forward(x)",
                    setup=None,
                    globals={"x": x, "fine_grained_cl": fine_grained_cl},
                    label="Fine grained + structured sparsity",
                    sub_label=sub_label,
                    description=(
                        "Fine-grained + structured sparsity @ " f"{sparsity}"
                    ),
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )
            results.append(
                benchmark.Timer(
                    stmt="mod(x)",
                    setup=None,
                    globals={"x": x, "mod": mod},
                    label="Dense benchmark",
                    sub_label=sub_label,
                    description="Dense benchmark",
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
            )

    compare = benchmark.Compare(results)
    f_name = "benchmark_v2_gpu.pkl"
    if device == "cpu":
        f_name = "benchmark_v2_gpu.pkl"
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

    for d in ["cpu", "gpu"]:
        if d == "cpu":
            device = torch.device("cpu")
            cuda = True
        else:
            device = torch.device("cuda")
            cuda = False
        mods, sparsities, = (
            [],
            [],
        )

        for sparsity, run_id in __RUN_IDS.items():
            mod = get_mod(run_id, device)
            mods.append(mod)
            sparsities.append(sparsity)
            num_features = mod.weight.shape[1]
            dtype = torch.float32
            num_threads = 1
        results = main(
            mods, sparsities, device, cuda, num_features, dtype, num_threads
        )


# NOTE: May play with the fuzzer again, so leaving this commented out as ref.
# fuzzer = benchmark.Fuzzer(
#     parameters=[
#         benchmark.FuzzedParameter("batch", distribution=batch_dist),
#         # benchmark.FuzzedParameter(
#         #     "k0", minval=64, maxval=10000, distribution="loguniform"
#         # ),
#         benchmark.FuzzedParameter(
#             "k0", minval=1024, maxval=1024, distribution="loguniform"
#         ),
#     ],
#     tensors=[
#         benchmark.FuzzedTensor(
#             "x",
#             size=("batch", "k0"),
#             min_elements=128,
#             max_elements=10000000,
#             cuda=cuda,
#             dtype=dtype,
#         )
#     ],
#     seed=42,
# )

# NOTE: May play with the fuzzer again, so leaving this commented out as ref
# for tensors, tensor_params, params in fuzzer.take(50):
# sub_label = f"{params['batch']:<6} x {params['k0']:<4}"
# for s in sparsities:
#     lc_params = dict(
#         in_features=int((1 - s) * tensors["x"].shape[1]),
#         out_features=10,
#         bias=True,
#         input_len=tensors["x"].shape[1],
#         fan_out_const=True,
#         device=torch.device(device),
#         dtype=dtype,
#     )
#     lc = LinearCondensed(**lc_params)
#     results.append(
#         benchmark.Timer(
#             stmt="lc_benchmark(x, lc)",
#             setup="from __main__ import lc_benchmark",
#             globals={"x": tensors["x"], "lc": lc},
#             label="Linear Condensed",
#             sub_label=sub_label,
#             description=f"Linear Condensed @ sparsity {s}",
#             num_threads=num_threads,
#         ).blocked_autorange(min_run_time=1)
#     )
# linear_params = dict(
#     in_features=tensors["x"].shape[1],
#     out_features=10,
#     bias=True,
#     device=torch.device(device),
#     dtype=dtype,
# )
# linear = torch.nn.Linear(**linear_params)
# results.append(
#     benchmark.Timer(
#         stmt="linear_benchmark(x, linear)",
#         setup="from __main__ import linear_benchmark",
#         globals={"x": tensors["x"], "linear": linear},
#         label="Linear",
#         sub_label=sub_label,
#         description="Linear",
#         num_threads=num_threads,
#     ).blocked_autorange(min_run_time=1)
# )
