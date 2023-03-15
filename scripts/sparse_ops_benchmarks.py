import torch
import torch.nn as nn
from torch.utils import benchmark
import pickle
from itertools import product
from sparseprop.utils import SparseLinear
from sparseprop.utils import swap_modules_with_sparse

from torch.nn.parallel import DistributedDataParallel
import pytorch_lightning as pl
import dotenv
import omegaconf


from rigl_torch.models import ModelFactory
from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.datasets import get_dataloaders
from rigl_torch.optim import (
    get_optimizer,
    get_lr_scheduler,
)
from rigl_torch.utils.rigl_utils import (
    get_T_end,
)
from hydra import initialize, compose
from rigl_torch.utils.sparse_ops import SparseModelFactory


@torch.no_grad()
def main(device, cuda, num_features, dtype, num_threads):
    sparsities = [0.0, 0.5, 0.9, 0.95, 0.99]
    batch_sizes = [2**x for x in range(10, -1, -1)]
    results = []
    counter = 0
    for batch_size, s in product(batch_sizes, sparsities):
        print(f"Benchmarking batch size {batch_size} with sparsity {s}...")
        counter += 1
        sub_label = f"{batch_size:<6} x {num_features:<4}"
        x = benchmark.FuzzedTensor(
            "x",
            size=(batch_size, num_features),
            # min_elements=2048,
            # max_elements=2048,
            cuda=cuda,
            dtype=dtype,
        )
        x = x.default_tensor_constructor(x._size, x._dtype)
        x = x.to(device=device)
        print(x.shape)
        if s != 0.0:
            linear_params = dict(
                in_features=x.shape[1],
                out_features=1000,
                bias=True,
                device=torch.device(device),
                dtype=dtype,
            )
            sparse_linear = nn.Linear(num_features, 1000)
            bias = (
                None
                if sparse_linear.bias is None
                else torch.nn.Parameter(sparse_linear.bias.data)
            )
            idx = torch.randperm(n=sparse_linear.weight.numel())
            non_zero_idx = idx[int(len(idx) * (1 - s)) :]
            w = sparse_linear.weight
            w = w.flatten()
            w[non_zero_idx] = 0
            w = w.reshape(sparse_linear.weight.shape)
            sparse_layer = SparseLinear(
                dense_weight=sparse_linear.weight.data, bias=bias
            )
            print(
                f"Testing sparsity == {1 - (sparse_linear.weight.count_nonzero() / sparse_linear.weight.numel())}"  # noqa
            )
            timer_kwargs = dict(
                label="Sparseprop",
                globals={"x": x, "layer": sparse_layer},
                description=f"Sparseprop @ sparsity {s}",
            )
        else:
            linear_params = dict(
                in_features=x.shape[1],
                out_features=1000,
                bias=True,
                device=torch.device(device),
                dtype=dtype,
            )
            linear = torch.nn.Linear(**linear_params)
            timer_kwargs = dict(
                label="Dense Linear",
                globals={"x": x, "layer": linear},
                description="Dense Linear",
            )

        results.append(
            benchmark.Timer(
                stmt="layer(x)",
                sub_label=sub_label,
                num_threads=num_threads,
                **timer_kwargs,
            ).blocked_autorange(min_run_time=5)
        )

    compare = benchmark.Compare(results)
    f_name = "compare_gpu_sparseprop.pkl"
    if device == "cpu":
        f_name = "compare_cpu_sparseprop.pkl"
    with open(f_name, "wb") as handle:
        pickle.dump(compare, handle)
    compare.colorize()
    print(compare)
    return results


def get_model_results(device, cuda, input_shape, dtype, num_threads):
    # sparsities = [0.0, 0.5, 0.9, 0.95, 0.99]
    sparsities = [0.0, 0.5]
    const_fan = [True, False]
    # batch_sizes = [2**x for x in range(10, -1, -1)]
    batch_sizes = [2**x for x in range(1, -1, -1)]
    results = []
    counter = 0
    for cf in const_fan:
        for s in sparsities:
            for batch_size in batch_sizes:
                print(
                    f"Benchmarking batch size {batch_size} with sparsity {s}..."
                )
                counter += 1
                x = benchmark.FuzzedTensor(
                    "x",
                    size=(batch_size, *input_shape),
                    cuda=cuda,
                    dtype=dtype,
                )
                x = x.default_tensor_constructor(x._size, x._dtype)
                x = x.to(device=device)
                sub_label = f"{x.shape:<6}"
                fully_sparse_model = get_model(
                    s, input_shape=x.shape, const_fan=cf, partial_sparsity=False
                )
                if s != 0.0:
                    partially_sparse_model = get_model(
                        s,
                        input_shape=x.shape,
                        const_fan=cf,
                        partial_sparsity=True,
                    )
                else:
                    partially_sparse_model = None
                timer_kwargs = []
                if s != 0.0:
                    fully_sparse_timer_kwargs = dict(
                        label=(
                            f"Fully Sparse ResNet50 with const_fan=={const_fan}"
                        ),
                        globals={"x": x, "layer": fully_sparse_model},
                        description=f"Fully Sparse ResNet50 @ sparsity {s}",
                    )
                    partially_sparse_timer_kwargs = dict(
                        label=(
                            f"Partially Sparse ResNet50 with const_fan=={const_fan}"  # noqa
                        ),
                        globals={"x": x, "layer": partially_sparse_model},
                        description=f"Partially Sparse ResNet50 @ sparsity {s}",
                    )
                    timer_kwargs.append(fully_sparse_timer_kwargs)
                    timer_kwargs.append(partially_sparse_timer_kwargs)
                else:
                    timer_kwargs.append(
                        dict(
                            label="Dense Linear",
                            globals={"x": x, "layer": fully_sparse_model},
                            description="Dense Linear",
                        )
                    )
                for k in timer_kwargs:
                    results.append(
                        benchmark.Timer(
                            stmt="layer(x)",
                            sub_label=sub_label,
                            num_threads=num_threads,
                            **k,
                        ).blocked_autorange(min_run_time=5)
                    )
                del fully_sparse_model
                del partially_sparse_model

        compare = benchmark.Compare(results)
        f_name = "compare_gpu_sparseprop_resnet50.pkl"
        if device == "cpu":
            f_name = "compare_cpu_sparseprop_resnet50.pkl"
        with open(f_name, "wb") as handle:
            pickle.dump(compare, handle)
        compare.colorize()
        print(compare)
        return results


def get_model(sparsity, input_shape, const_fan, partial_sparsity):
    with initialize("../configs", version_base="1.2.0"):
        dense_alloc = 1 - sparsity
        # print(f"Dense alloc: {dense_alloc}")
        if int(dense_alloc) == 1:
            dense_alloc = "null"
        cfg = compose(
            "config.yaml",
            overrides=[
                "dataset=imagenet",
                "compute.distributed=False",
                "model=resnet50",
                f"rigl.dense_allocation={dense_alloc}",
                f"rigl.const_fan_in={const_fan}",
            ],
        )
    rank = 0
    _, optimizer_state, scheduler_state, pruner_state, _ = (
        None,
        None,
        None,
        None,
        None,
    )

    if "diet" not in cfg.rigl:
        with omegaconf.open_dict(cfg):
            cfg.rigl.diet = None
    if "keep_first_layer_dense" not in cfg.rigl:
        with omegaconf.open_dict(cfg):
            cfg.rigl.keep_first_layer_dense = False

    pl.seed_everything(cfg.training.seed)
    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_dataloaders(cfg)

    model = ModelFactory.load_model(
        model=cfg.model.name, dataset=cfg.dataset.name, diet=cfg.rigl.diet
    )
    model.to(device)
    if cfg.compute.distributed:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = get_optimizer(cfg, model, state_dict=optimizer_state)
    _ = get_lr_scheduler(cfg, optimizer, state_dict=scheduler_state)
    _ = None
    # print(f"dense alloc is: {cfg.rigl.dense_allocation}")
    # print(f"of type: {type(cfg.rigl.dense_allocation)}")
    if cfg.rigl.dense_allocation is not None:
        T_end = get_T_end(cfg, [0 for _ in range(0, 1251)])
        if cfg.rigl.const_fan_in:
            rigl_scheduler = RigLConstFanScheduler
        else:
            rigl_scheduler = RigLScheduler
        _ = rigl_scheduler(
            model,
            optimizer,
            dense_allocation=cfg.rigl.dense_allocation,
            alpha=cfg.rigl.alpha,
            delta=cfg.rigl.delta,
            static_topo=cfg.rigl.static_topo,
            T_end=T_end,
            ignore_linear_layers=cfg.rigl.ignore_linear_layers,
            grad_accumulation_n=cfg.rigl.grad_accumulation_n,
            sparsity_distribution=cfg.rigl.sparsity_distribution,
            erk_power_scale=cfg.rigl.erk_power_scale,
            state_dict=pruner_state,
            filter_ablation_threshold=cfg.rigl.filter_ablation_threshold,
            static_ablation=cfg.rigl.static_ablation,
            dynamic_ablation=cfg.rigl.dynamic_ablation,
            min_salient_weights_per_neuron=cfg.rigl.min_salient_weights_per_neuron,  # noqa
            use_sparse_init=cfg.rigl.use_sparse_initialization,
            init_method_str=cfg.rigl.init_method_str,
            use_sparse_const_fan_in_for_ablation=cfg.rigl.use_sparse_const_fan_in_for_ablation,  # noqa
            initialize_grown_weights=cfg.rigl.initialize_grown_weights,
        )
    if cfg.rigl.dense_allocation is not None:
        if partial_sparsity:
            model = swap_modules_with_sparse(
                model, input_shape=input_shape, inplace=True
            )
        else:
            model = SparseModelFactory().get_sparse_model(
                model.to("cpu"), input_shape=input_shape
            )
    return model


if __name__ == "__main__":
    dotenv.load_dotenv("../.env")
    num_features = 1000
    dtype = torch.float32
    cuda = False
    dtype = torch.float32
    num_threads = 1
    d = "cpu"
    # results = main(d, cuda, num_features, dtype, num_threads)

    imagenet_input_shape = (3, 224, 224)
    model_results = get_model_results(
        d, cuda, imagenet_input_shape, dtype, num_threads
    )
