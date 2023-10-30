from torch import _dynamo as dynamo
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
import gc

from condensed_sparsity.condensed_linear import (  # noqa
    CSRLinear,
    CondensedLinearStructured,
    CondensedLinearFineGrained,
    VmapCondensed,
    CondensedLinearFineGrainedSparseOp,  # noqa
)

__MIN_RUN_TIME = 2


@torch.no_grad()
def main(
    mods: List[nn.Module],
    sparsities: List[float],
    device,
    cuda,
    num_features,
    dtype,
    num_threads,
    compiler,
    compiler_kwargs,
    include_csr=True,
    skip_eager=False,
    include_vmap=True,
    seq_len=197,
):
    __DISABLED_BACKENDS = ["tvm", "onnxrt"]
    # NOTE: tvm has issues with NoneType resolution from index slice operator in
    # fine-grain condensed. Unknown issue in onnxrt.

    batch_sizes = [2**x for x in range(11, -1, -1)]
    results = []
    counter = 0
    for mod, sparsity in zip(mods, sparsities):
        label = (
            f"Sparsity {sparsity} with {num_threads} threads "
            f"using compilation strategy {compiler} "
            f"and dtype {dtype} on device {device}."
        )

        # Get condensed modules
        mod = mod.type(dtype)
        mod.eval()
        cl_struc = CondensedLinearStructured(deepcopy(mod), dtype=dtype).eval()
        cl_fine = CondensedLinearFineGrained(deepcopy(mod), dtype=dtype).eval()
        if include_vmap:
            cl_vmap = VmapCondensed(deepcopy(mod), dtype=dtype).eval()
        if include_csr:
            cl_sparse_op = CondensedLinearFineGrainedSparseOp(
                deepcopy(mod), dtype=dtype
            ).eval()
            csr_linear = CSRLinear(deepcopy(mod), dtype=dtype).eval()
        else:
            cl_sparse_op = None
            csr_linear = None

        for batch_size in batch_sizes:
            sub_label = f"{batch_size:<6} x {seq_len:<3} x {num_features:<4}"
            print(f"Benchmarking {sub_label}...")
            counter += 1
            # Load input tensor
            # 197 is seq length for (224/16)**2 + 1
            # (i.e., imagenet 16x16 patches + start token)
            x = benchmark.FuzzedTensor(
                "x",
                size=(batch_size, seq_len, num_features),
                cuda=cuda,
                dtype=dtype,
                probability_contiguous=1.0,
            )
            x = x.default_tensor_constructor(x._size, x._dtype)
            x = x.to(device=device)

            # First we benchmark uncompiled
            if not skip_eager:
                with torch.no_grad():
                    # Uncompiled benchmarks
                    ## Eager dense benchmark
                    _ = mod(x)
                    results.append(
                        benchmark.Timer(
                            stmt="mod(x)",
                            setup="",
                            globals={"x": x, "mod": mod},
                            label=label,
                            sub_label=sub_label,
                            description="Dense benchmark - eager",
                            num_threads=num_threads,
                        ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                    )

                    # Structured
                    _ = cl_struc(x)  # JIT warmup and caching
                    results.append(
                        benchmark.Timer(
                            stmt="cl_struc(x)",
                            setup="",
                            globals={"x": x, "cl_struc": cl_struc},
                            label=label,
                            sub_label=sub_label,
                            description=("Structured sparsity - eager"),
                            num_threads=num_threads,
                        ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                    )

                    # csr eager
                    if include_csr:
                        _ = cl_sparse_op(x)
                        results.append(
                            benchmark.Timer(
                                stmt="cl_sparse_op(x)",
                                setup="",
                                globals={"x": x, "cl_sparse_op": cl_sparse_op},
                                label=label,
                                sub_label=sub_label,
                                description=("structured + csr - eager"),
                                num_threads=num_threads,
                            ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        )

                        _ = csr_linear(x)
                        results.append(
                            benchmark.Timer(
                                stmt="csr_linear(x)",
                                setup="",
                                globals={"x": x, "csr_linear": csr_linear},
                                label=label,
                                sub_label=sub_label,
                                description=("csr only - eager"),
                                num_threads=num_threads,
                            ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        )

                    # Vmap eager
                    if include_vmap:
                        _ = cl_vmap(x)
                        results.append(
                            benchmark.Timer(
                                stmt="cl_vmap(x)",
                                setup="",
                                globals={"x": x, "cl_vmap": cl_vmap},
                                label=label,
                                sub_label=sub_label,
                                description=("Vmap - eager"),
                                num_threads=num_threads,
                            ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        )

                    ## Eager fine grained
                    _ = cl_fine(x)
                    results.append(
                        benchmark.Timer(
                            stmt="cl_fine(x)",
                            setup="",
                            globals={
                                "x": x,
                                "cl_fine": cl_fine,
                            },
                            label=label,
                            sub_label=sub_label,
                            description=("Fine-grained + structured - eager"),
                            num_threads=num_threads,
                        ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                    )

            # Compiled/jit benchmarks
            if compiler is not None:
                # for backend in dynamo.list_backends():
                for backend in ["inductor"]:
                    if backend in __DISABLED_BACKENDS:
                        continue
                    if compiler in ["trace", "script"]:
                        backend = compiler
                    dynamo.reset()

                    # Compilation
                    if compiler == "trace":
                        cl_struct_compiled = torch.jit.trace(cl_struc, x)
                        cl_fine_compiled = torch.jit.trace(cl_fine, x)
                        mod_compiled = torch.jit.trace(mod, x)
                        # vmap_compiled = torch.jit.trace(cl_vmap, x)
                        if include_csr:
                            cl_sparse_op_compiled = torch.jit.trace(
                                cl_sparse_op, x
                            )
                            csr_compiled = torch.jit.trace(csr_linear, x)

                    elif compiler == "script":
                        cl_struct_compiled = torch.jit.optimize_for_inference(
                            torch.jit.freeze(torch.jit.script(cl_struc, x))
                        )
                        cl_fine_compiled = torch.jit.optimize_for_inference(
                            torch.jit.freeze(torch.jit.script(cl_fine, x))
                        )
                        mod_compiled = torch.jit.optimize_for_inference(
                            torch.jit.freeze(torch.jit.script(mod, x))
                        )
                        # vmap_compiled = torch.jit.optimize_for_inference(
                        #     torch.jit.script(cl_vmap, x)
                        # )
                        if include_csr:
                            cl_sparse_op_compiled = (
                                torch.jit.optimize_for_inference(
                                    torch.jit.freeze(
                                        torch.jit.script(cl_sparse_op, x)
                                    )
                                )
                            )
                            csr_compiled = torch.jit.trace(
                                torch.jit.optimize_for_inference(
                                    torch.jit.freeze(
                                        torch.jit.script(csr_linear, x)
                                    )
                                )
                            )

                    elif compiler == "inductor":
                        cl_struct_compiled = torch.compile(
                            cl_struc, **compiler_kwargs
                        )
                        cl_fine_compiled = torch.compile(
                            cl_fine, backend=backend, **compiler_kwargs
                        )
                        mod_compiled = torch.compile(
                            mod, backend=backend, **compiler_kwargs
                        )
                        # vmap_compiled = torch.compile(
                        #     cl_vmap, backend=backend, **compiler_kwargs
                        # )
                        if include_csr:
                            cl_sparse_op_compiled = torch.compile(
                                cl_sparse_op, mode=compiler_kwargs["mode"]
                            )
                            csr_compiled = torch.compile(
                                csr_linear, mode=compiler_kwargs["mode"]
                            )

                    # Explanations of compiling for debugging
                    # (
                    #     explanation,
                    #     out_guards,
                    #     graphs,
                    #     ops_per_graph,
                    #     break_reasons,
                    #     explanation_verbose,
                    # ) = dynamo.explain(mod.forward, x)
                    # print(explanation_verbose)
                    # print(ops_per_graph)
                    # *_, explanation_verbose = dynamo.explain(cl_struc.forward, x)  # noqa
                    # print(explanation_verbose)
                    # *_, explanation_verbose = dynamo.explain(cl_fine.forward, x)  # noqa
                    # print(explanation_verbose)
                    # exit()

                    # Compilation
                    with torch.no_grad():
                        # Compiled dense benchmark
                        _ = mod_compiled(x)
                        results.append(
                            benchmark.Timer(
                                stmt="mod_compiled(x)",
                                setup="",
                                globals={"x": x, "mod_compiled": mod_compiled},
                                label=label,
                                sub_label=sub_label,
                                description=(
                                    "Dense benchmark - Compiled - backend "
                                    f"{backend}"
                                ),
                                num_threads=num_threads,
                            ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        )
                        # Structured compiled
                        _ = cl_struct_compiled(x)  # JIT warmup and caching
                        results.append(
                            benchmark.Timer(
                                stmt="cl_struct_compiled(x)",
                                setup="",
                                globals={
                                    "x": x,
                                    "cl_struct_compiled": cl_struct_compiled,
                                },
                                label=label,
                                sub_label=sub_label,
                                description=(
                                    "Structured sparsity compiled "
                                    f"with backend - {backend}"
                                ),
                                num_threads=num_threads,
                            ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        )
                        ## CSR Sparse Format
                        if include_csr:
                            _ = cl_sparse_op_compiled(x)
                            results.append(
                                benchmark.Timer(
                                    stmt="cl_sparse_op_compiled(x)",
                                    setup="",
                                    globals={
                                        "x": x,
                                        "cl_sparse_op_compiled": cl_sparse_op_compiled,  # noqa
                                    },
                                    label=label,
                                    sub_label=sub_label,
                                    description=(
                                        "structured + csr with backend "
                                        f"{backend}"
                                    ),
                                    num_threads=num_threads,
                                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                            )
                            _ = csr_compiled(x)
                            results.append(
                                benchmark.Timer(
                                    stmt="csr_compiled(x)",
                                    setup="",
                                    globals={
                                        "x": x,
                                        "csr_compiled": csr_compiled,
                                    },
                                    label=label,
                                    sub_label=sub_label,
                                    description=(
                                        "csr only compiled with backend - "
                                        f"{backend}"
                                    ),
                                    num_threads=num_threads,
                                ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                            )
                        # Fine grained compiled
                        _ = cl_fine_compiled(x)
                        results.append(
                            benchmark.Timer(
                                stmt="cl_fine_compiled(x)",
                                setup="",
                                globals={
                                    "x": x,
                                    "cl_fine_compiled": cl_fine_compiled,
                                },
                                label=label,
                                sub_label=sub_label,
                                description=(
                                    "Fine-grained + structured with "
                                    f"backend {backend}"
                                ),
                                num_threads=num_threads,
                            ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        )
                        # Vmap benchmarks
                        # _ = vmap_compiled(x)
                        # results.append(
                        #     benchmark.Timer(
                        #         stmt="vmap_compiled(x)",
                        #         setup="",
                        #         globals={
                        #             "x": x,
                        #             "vmap_compiled": vmap_compiled,
                        #         },
                        #         label=label,
                        #         sub_label=sub_label,
                        #         description=(
                        #             "Vmap - Compiled - backend " f"{backend}"
                        #         ),
                        #         num_threads=num_threads,
                        #     ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
                        # )

                        ## Clean up compiled
                        del mod_compiled
                        del cl_struct_compiled
                        del cl_fine_compiled
                        if include_csr:
                            del cl_sparse_op_compiled
                            del csr_compiled
                        # del vmap_compiled
                        gc.collect()

                    if compiler in ["trace", "script"]:
                        break  # No need to run through other backends

        # clean up eager modules
        del cl_struc
        del cl_fine
        if include_vmap:
            del cl_vmap
        if include_csr:
            del cl_sparse_op
            del csr_linear

    # Collate results and save
    compare = benchmark.Compare(results)
    if not cuda:
        device_name = "cpu"
    else:
        device_name = "gpu"
    f_name = (
        f"benchmark_3dim_{device_name}_threads_{num_threads}_"
        f"compiler_{compiler}_dtype_{dtype}final_hector_nice-15.pkl"
    )
    with open(f_name, "wb") as handle:
        pickle.dump(compare, handle)
    compare.colorize()
    print(compare)
    return results


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
    os.environ["IMAGE_NET_PATH"]
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
    # import logging

    dynamo.config.verbose = True
    # dynamo.config.log_level = logging.DEBUG
    # dynamo.config.output_code = True

    compiler_kwargs = {
        "mode": "max-autotune",
        "fullgraph": True,
    }
    # torch.set_float32_matmul_precision("high")
    # NOTE: Below results in errors during compilation
    # torch.jit.enable_onednn_fusion(
    #     True
    # )  # seems like we need this to use inductor on gpu
    __RUN_IDS = {
        99: "20230911_3f1ffqmr",
        95: "20230911_1mxhel1q",
        90: "20230601_nrblbn15",
        80: "20230601_0p0wrlb0",
    }
    # for dtype in [torch.float32, torch.bfloat16]:
    # for num_threads in [1, 40, 80]:
    __LAYER_NAME = "encoder.layers.encoder_layer_11.mlp.3"
    for num_threads in [1, 2, 4, 8, 16]:
        for compiler in [
            "inductor",
            # "script",
            # "trace",
        ]:
            for d in [
                "cpu",
                # "gpu"
            ]:  # NOTE: Need gpu runs for bf16 all threads
                if d == "cpu":
                    device = torch.device("cpu")
                    cuda = False
                else:
                    device = torch.device("cuda")
                    cuda = True
                mods, sparsities = [], []

                for sparsity, run_id in __RUN_IDS.items():
                    mod = get_mod(run_id, device, __LAYER_NAME)
                    mod.to(device)
                    mods.append(mod)
                    sparsities.append(sparsity)
                    num_features = mod.weight.shape[1]
                    dtype = (
                        # torch.bfloat16
                        # bfloat16 not supported for CSR sparse
                        # float16 not supported by IPEX
                        torch.float32
                    )  # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html  # noqa
                    if dtype == torch.bfloat16 or compiler in [
                        "script",
                        "trace",
                    ]:
                        include_csr = False  # Doesn't work with above options
                    else:
                        include_csr = True
                results = main(
                    mods,
                    sparsities,
                    device,
                    cuda,
                    num_features,
                    dtype,
                    num_threads,
                    compiler,
                    compiler_kwargs,
                    include_csr,
                    skip_eager=True,
                    include_vmap=False,
                )

# @jax.jit
# def sp2(X, weights):
#  xx = jnp.take(X, indices, axis=1)
# return jnp.einsum('abc,bc->ab', xx, weights)
# in_size = 2000
# out_size = 2000
# batch_size = 32
# n_active = int(in_size * 0.05)
# X = random.uniform(key1, (batch_size, in_size))
# W_dense = random.uniform(key2, (in_size, out_size))
# # This defines a sparse matrix with fixed number of non-zeros at every row.
# W_sparse = W_dense[:n_active, :]
# indices = random.randint(key2, minval=0, maxval=matrix_size, shape=(n_active, out_size))  # noqa
