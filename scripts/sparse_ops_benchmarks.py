import torch
import torch.nn as nn
from torch.utils import benchmark
import pickle
from itertools import product
from sparseprop.utils import SparseLinear


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


if __name__ == "__main__":
    num_features = 1000
    dtype = torch.float32
    cuda = False
    dtype = torch.float32
    num_threads = 1
    d = "cpu"
    results = main(d, cuda, num_features, dtype, num_threads)
