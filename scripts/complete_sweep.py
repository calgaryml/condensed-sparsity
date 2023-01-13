import subprocess
import pickle

with open("missing_sweep_params.pkl", "rb") as handle:
    missing_params = pickle.load(handle)

for instance in missing_params:
    args = [f"{k}={v}" for k, v in instance.items() if k != "sweep_id"]
    arg_string = "./slurm/manual_missing_sweeps.sh "
    config_string = (
        "dataset=cifar10 "
        "rigl.delta=100 "
        "rigl.grad_accumulation_n=1 "
        "training.batch_size=128 "
        "training.max_steps=null "
        "training.weight_decay=5.0e-4 "
        "training.label_smoothing=0 "
        "training.lr=0.1 "
        "training.epochs=250 "
        "training.warm_up_steps=0 "
        "training.scheduler=step_lr "
        "training.step_size=77 "
        "training.gamma=0.2 "
        "compute.distributed=False "
        "rigl.use_sparse_initialization=True "
        "rigl.init_method_str=grad_flow_init "
    )
    for x in args:
        config_string += f"{x} "
    command = (arg_string + config_string).strip().split(" ")
    print(command)
    # command = (arg_string+config_string)
    # subprocess.run([arg_string, config_string])
    subprocess.run(command)
    break
