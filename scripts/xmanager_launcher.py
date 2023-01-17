# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XManager launcher that runs an image built from a Dockerfile."""

from dotenv import dotenv_values
import copy  # noqa
from typing import Sequence

from absl import app
from xmanager import xm
from xmanager import xm_local

_cifar_args = [
    "training.dry_run=True",  # TODO
    "dataset=cifar10",
    "model=resnet18",
    "rigl.dense_allocation=0.01",
    "rigl.delta=100",
    "rigl.grad_accumulation_n=1",
    "rigl.min_salient_weights_per_neuron=0.3",
    "training.batch_size=128",
    "training.max_steps=null",
    "training.weight_decay=5.0e-4",
    "training.label_smoothing=0",
    "training.lr=0.1",
    "training.epochs=250",
    "training.warm_up_steps=0",
    "training.scheduler=step_lr",
    "training.gamma=0.2",
    "compute.distributed=False",
    "training.step_size=77",
]

_imagenet_args = [
    "training.dry_run=True",  # TODO
    "dataset=imagenet",
    "model=resnet50",
    "rigl.dense_allocation=0.1",
    "rigl.delta=800",
    "rigl.grad_accumulation_n=8",
    "rigl.min_salient_weights_per_neuron=0.3",
    "training.batch_size=512",
    "training.max_steps=256000",
    "training.weight_decay=0.0001",
    "training.label_smoothing=0.1",
    "training.lr=0.2",
    "training.epochs=104",
    "training.warm_up_steps=5",
    "training.scheduler=step_lr_with_warm_up",
    "training.gamma=0.1",
    "compute.distributed=True",
    "compute.world_size=2",
    "rigl.use_sparse_initialization=True",
    "rigl.init_method_str=grad_flow_init",
]
_large_batch_imagenet_args = [
    "dataset=imagenet",
    "model=resnet50",
    "rigl.dense_allocation=0.1",  # TODO
    "rigl.delta=100",
    "rigl.grad_accumulation_n=1",
    "rigl.min_salient_weights_per_neuron=0.3",
    "training.batch_size=4096",
    "training.max_steps=160000",
    "training.weight_decay=0.0001",
    "training.label_smoothing=0.1",
    "training.lr=1.6",
    "training.epochs=515",
    "training.warm_up_steps=25",
    "training.scheduler=step_lr_with_warm_up",
    "training.gamma=0.1",
    "compute.distributed=True",
    "compute.world_size=12",
    "rigl.use_sparse_initialization=True",
    "rigl.init_method_str=grad_flow_init",
]


def main(argv: Sequence[str]) -> None:
    del argv
    docker_image = "mklasby/condensed-sparsity:rigl_gcs"
    # docker_image = "gcr.io/external-collab-experiment/condensed_sparsity:20230116-211607-665612"  # noqa
    with xm_local.create_experiment(
        experiment_title="condensed-sparsity-docker-arg-test"
    ) as experiment:
        executable_spec = xm.Dockerfile(
            path="/home/mike/condensed-sparsity/",
            dockerfile="/home/mike/condensed-sparsity/Dockerfile.gcs",
        )
        executable_spec = xm.Container(docker_image)
        env_vars = dotenv_values("/home/mike/condensed-sparsity/.env.gcs")

        [executable] = experiment.package(
            [
                xm.Packageable(
                    executable_spec=executable_spec,
                    executor_spec=xm_local.Vertex.Spec(),
                    args={},
                    env_vars=env_vars,
                ),
            ]
        )
        env_vars = dotenv_values("/home/mike/condensed-sparsity/.env.gcs")
        args = ["python", "train_rigl.py"]
        # args.extend(_cifar_args)
        # for dense_alloc in [0.01, 0.05, 0.1, 0.2]:
        #     these_args = copy.deepcopy(args)
        #     these_args.extend([f"rigl.dense_allocation={dense_alloc}"])
        # executor = xm_local.Vertex(xm.JobRequirements(t4=1))

        # args.extend(_imagenet_args)
        # executor=xm_local.Vertex(xm.JobRequirements(a100=2))

        args.extend(_large_batch_imagenet_args)
        executor = xm_local.Vertex(xm.JobRequirements(a100=12))
        experiment.add(
            xm.Job(
                executable=executable,
                executor=executor,
                env_vars=env_vars,
                args=args,
            )
        )
        exit()


if __name__ == "__main__":
    app.run(main)
