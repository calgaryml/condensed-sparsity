import torch
from torch.utils import benchmark
import pickle
from itertools import product

from condensed_sparsity.models import LinearCondensed


def lc_benchmark(x, lc):
    lc(x)
    return


def linear_benchmark(x, linear):
    linear(x)
    return


@torch.no_grad()
def main(device, cuda, num_features, dtype, num_threads):
    sparsities = [0.0, 0.5, 0.9, 0.95, 0.99]
    batch_sizes = [2**x for x in range(9, -1, -1)]
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
            lc_params = dict(
                in_features=int((1 - s) * x.shape[1]),
                out_features=1000,
                bias=True,
                input_len=x.shape[1],
                fan_out_const=False,
                device=torch.device(device),
                dtype=dtype,
            )
            lc = LinearCondensed(**lc_params)
            jit_lc = torch.jit.trace(lc, x)
            results.append(
                benchmark.Timer(
                    stmt="lc_benchmark(x, lc)",
                    setup="from __main__ import lc_benchmark",
                    globals={"x": x, "lc": jit_lc},
                    label="Linear Condensed",
                    sub_label=sub_label,
                    description=f"Linear Condensed @ sparsity {s}",
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=5)
            )
        else:
            linear_params = dict(
                in_features=x.shape[1],
                out_features=10,
                bias=True,
                device=torch.device(device),
                dtype=dtype,
            )
            linear = torch.nn.Linear(**linear_params)
            results.append(
                benchmark.Timer(
                    stmt="linear_benchmark(x, linear)",
                    setup="from __main__ import linear_benchmark",
                    globals={"x": x, "linear": linear},
                    label="Linear",
                    sub_label=sub_label,
                    description="Linear",
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=5)
            )

    compare = benchmark.Compare(results)
    f_name = "compare_gpu.pkl"
    if device == "cpu":
        f_name = "compare_cpu_jit.pkl"
    with open(f_name, "wb") as handle:
        pickle.dump(compare, handle)
    compare.colorize()
    print(compare)
    return results


if __name__ == "__main__":
    # for d in ["cuda:0", "cpu"]:
    for d in ["cpu"]:
        cuda = True
        # num_features = 65536
        num_features = 20000
        dtype = torch.float32
        num_threads = 1
        if d == "cpu":
            cuda = False
            dtype = torch.float32
            num_threads = 1
        results = main(d, cuda, num_features, dtype, num_threads)


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
