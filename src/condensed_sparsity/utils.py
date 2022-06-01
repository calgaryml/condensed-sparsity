import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os
import sys
# import custom_datasets as cu_datasets


def make_outputdirname(args):

    output_dir = f"output/{args.model_type}{args.num_layers}_{args.dataset}"
    make_linear = args.make_linear == "True"
    act_fctn = "Linear" if make_linear else "ReLU"
    bias_tag = "_no_bias" if args.no_bias else ""

    if args.dataset == "svhn_resnet18":
        input_size = 512
        output_size = 10
    elif args.dataset == "MNIST":
        input_size = 28**2
        output_size = 10
    else:
        input_size = "NA"
        output_size = "NA"

    if args.model_type == "simpleCNN":
        output_dir += f"_conv_{args.num_out_conv}_fcfanin_{args.fan_in}_fcout_{args.num_out_fc}"
    else:
        output_dir += f"_n1_{input_size}"
        if args.num_layers > 2:
            output_dir += f"_nmid_{args.num_mid}_k_{args.fan_in}"
        output_dir += f"_n{args.num_layers}_{output_size}"
        # if args.individ_indx_seqs: output_dir+=f'_individ_indx_seqs'
        if args.model_type == "SparseNet":
            output_dir += f"_{args.sparsity_type}_{args.connect_type}"

    output_dir += f"_{act_fctn}{bias_tag}_fanout_const_{args.fan_out_const}"

    if args.train_subset_size > 0:
        output_dir += f"_train_on_{args.train_subset_size}_samples"
    if args.normalize_pixelwise:
        output_dir += f"_pixelwise_normalization"

    output_dir += f"_LR_{args.lr}"
    if args.lr_decay != "none":
        output_dir += f"_{args.lr_decay}_LRdecay"
    output_dir += f"_mbs_{args.mbs}"

    output_dir += f"_seed_{args.seed}"

    return output_dir


def spe_matmul(inputs, weights, indx_seqs):

    """
    Performs special matrix multiplication.

    Args:
        inputs:  2d array, batch of input vectors, shape (batch_size, input_len)
        weights: 2d array, weight matrix, shape (num_units, fan_in)
        indx_seqs: 2d array, sequences of input vector indices, shape (num_units, fan_in)

    Returns:
        Layer output, shape (batch_size, num_units)
    """

    assert indx_seqs.shape == weights.shape

    # for each set of recombination vectors:
    # element-wise multiplication along the rows and summation of all vectors
    # --> output has shape (batch_size, num_units)
    v_out = torch.sum(weights * inputs[:, indx_seqs], axis=2)

    return v_out


def gen_indx_seqs(num_in, num_out, input_len, fan_out_const):
    """
    Generates indices for drawing recombination vectors from the input vector v.

        number recomb.vectors = num_in (= fan_in, also called k)
        length of each recomb.vector = num_out (= n2)

    Args:
        num_out, num_in= CondLayer.weight.shape
        input_len: int, length of the input vector
        fan_out_const: bool, if True, nearly constant fan-out will be ensured

    Returns:
        A 2d array of indices of the same shape as the weight matrix.
    """

    # init index sequences
    # indx_seqs= np.zeros((weights.shape))
    indx_seqs = np.zeros((num_out, num_in))

    # indices of v (the input of length d)
    v_inds = np.arange(input_len)

    # initialize an array of probabs for every index of v (initially uniform)
    probs = 1 / input_len * np.ones(input_len)

    for row_nr in range(num_out):
        chosen_inds = np.random.choice(
            v_inds, size=num_in, replace=False, p=probs / sum(probs)
        )
        chosen_inds.sort()
        # update probabs only if want to control fan_out
        if fan_out_const:
            probs[chosen_inds] /= 100 * input_len

        indx_seqs[row_nr, :] = chosen_inds

    return indx_seqs.astype(int)


def make_smask(dims, fan_in, sparsity_type, connect_type, fan_out_const):
    """
    dims: layer dimensions, as layer.weight.shape

    fan_in: number of incoming connections per neuron;
        exact if sparsity_type=='per_neuron'
        average if sparsity_type=='per_layer'

    sparsity_type:
        'per_neuron'= pruning per neuron, such that each neuron has exactly fan_in connections
        'per_layer' = pruning per layer, such that each neuron has fan_in connections on average

    connect_type: applies only in case of sparsity_type=='per_neuron'; specifies how incoming connections are structured
        'scattered'= incoming connections chosen randomly
        'block'    = incoming connections from adjacent neurons

    fan_out_const: bool; if True, will ensure that fan_out is (nearly) constant (depending on the layer dims and fan_in, an exactly constant fan_out is not always possible)

    * *

    The function gen_indx_seqs samples elements for a 2d array
    from a given 1d array with certain restrictions in place.
    The frequency of individual elements can be controlled by
    weighting the draws for each element with a probability.

    Here, I create a sparsity pattern (mask) by sampling the elements of a 2d array from a uniform distrib, then applying the topk() function to select the top k among them and set them to 1 in the mask, rest to 0.

    In this seetting, fan_in/fan_out is the number of nonzero elements per row/column.
    When implementing per-neuron sparsity (i.e., constant fan-in),
    I apply topk() on each row. For constant fan-out,
    I need to ensure that the number of nonzero elements per col is const.
    For fan-in types "scattered" and "block".

    """
    device = torch.device("cpu")
    smask = torch.FloatTensor(dims, device=device).uniform_()
    dim_out, dim_in = dims
    # setting top num_to_freeze values in smask to 1;
    # the corresponding values in the weight tensor will be set to zero and frozen

    # ========================
    # ==== A) per-neuron sparsity (fan_in is exact per neuron)
    # ========================
    if sparsity_type == "per_neuron":
        #
        # == a) scattered-to-1 (fan-in connections chosen randomly)
        # ====================
        if connect_type == "scattered":
            if fan_out_const:
                sys.exit(
                    f"sparsity_type {sparsity_type} connect_type {connect_type} >>> fan_out_const does not work yet! TO BE DONE!"
                )
            else:
                ntf_per_neuron = dim_in - fan_in
                # select topk values in each row
                k_vals, k_idx = smask.topk(k=int(ntf_per_neuron), dim=1)
                # note: k_idx.shape = [dim_out, dim_in-fan_in]

                smask.zero_()
                smask[
                    torch.arange(dim_out)[:, None], k_idx
                ] = 1  # setting to 1 weights to be removed

        #
        # == b) block-to-1 (fan-in connections to adjacent neurons)
        # ================
        elif connect_type == "block":
            if fan_out_const:
                sys.exit(
                    f"sparsity_type {sparsity_type} connect_type {connect_type} >>> fan_out_const does not work yet! TO BE DONE!"
                )
            else:
                # draw dim_out start indices (1 for each block)
                k_idx = torch.randint(
                    low=0, high=dim_in - fan_in + 1, size=(dim_out, 1)
                )
                # add subsequent indices to cover full block
                k_idx = k_idx.repeat(1, fan_in)
                for i in range(fan_in):
                    k_idx[:, i] += i

                smask.fill_(1)  # setting 1s everywhere
                smask[
                    torch.arange(dim_out)[:, None], k_idx
                ] = 0  # setting to 0 where weights to keep
    #
    # =======================
    # ==== B) global per-layer sparsity (fan_in is average)
    # =======================
    elif sparsity_type == "per_layer":
        # dim_out is the number of neurons
        ntf_per_layer = (dim_in - fan_in) * dim_out
        r = torch.topk(smask.view(-1), ntf_per_layer)
        smask = torch.FloatTensor(dims, device=device).fill_(0)

        for i, v in zip(r.indices, r.values):
            index = i.item()
            i_col = index % dims[-1]
            i_row = index // dims[-1]
            smask[i_row, i_col] = 1

    smask = smask.to(bool)
    s = torch.sum(smask).item()
    p = 100 * s / np.prod(dims)
    print(
        f"Applying smask to layer will freeze {s} of {np.prod(dims)} weights (= {p:.2f}% sparse)."
    )

    return smask


def pixelwise_normalization(images):
    orig_type = type(images)
    if orig_type == torch.Tensor:
        images = images.numpy()

    # === compute the mean and stdev of each pixel across images
    pix_mean = np.mean(images, axis=0)
    pix_stdev = np.std(images, axis=0)

    # === normalize images pixel-wise
    a = images - pix_mean
    images = np.divide(
        a, pix_stdev, out=np.zeros(a.shape, dtype=float), where=pix_stdev != 0
    )

    if orig_type == torch.Tensor:
        images = torch.Tensor(images)

    return images


def load_dataset(
    dataset, dataset_dir, batch_size, test_batch_size, no_da, kwargs
):
    if dataset == "MNIST":
        train_loader = load_MNIST(
            data_dir=dataset_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            kwargs=kwargs,
        )
        test_loader = load_MNIST(
            data_dir=dataset_dir,
            split="test",
            batch_size=test_batch_size,
            shuffle=False,
            kwargs=kwargs,
        )
        input_size = 28**2
        num_classes = 10
        num_channels = 1
    elif dataset == "cifar10":
        train_loader = load_CIFAR10(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            no_da=no_da,
            kwargs=kwargs,
        )
        test_loader = load_CIFAR10(
            split="test",
            batch_size=test_batch_size,
            shuffle=False,
            no_da=no_da,
            kwargs=kwargs,
        )
        num_channels = 3
        input_size = 32**2 * num_channels
        num_classes = 10
    elif dataset == "FashionMNIST":
        train_loader = load_FashionMNIST(
            data_dir=dataset_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            kwargs=kwargs,
        )
        test_loader = load_FashionMNIST(
            data_dir=dataset_dir,
            split="test",
            batch_size=test_batch_size,
            shuffle=False,
            kwargs=kwargs,
        )
        input_size = 28**2
        num_classes = 10
        num_channels = 1
    elif dataset == "cifar100_resnet50":
        train_loader = load_cifar100_resnet50(
            data_dir=dataset_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            kwargs=kwargs,
        )
        test_loader = load_cifar100_resnet50(
            data_dir=dataset_dir,
            split="test",
            batch_size=test_batch_size,
            shuffle=False,
            kwargs=kwargs,
        )
        input_size = 2048
        num_classes = 100
        num_channels = 1
    elif dataset == "svhn_resnet18":
        train_loader = load_svhn_resnet18(
            data_dir=dataset_dir,
            split="train",
            batch_size=batch_size,
            shuffle=False,
            kwargs=kwargs,
        )
        test_loader = load_svhn_resnet18(
            data_dir=dataset_dir,
            split="test",
            batch_size=test_batch_size,
            shuffle=False,
            kwargs=kwargs,
        )
        input_size = 512
        num_classes = 10
        num_channels = 1
    else:
        print(f"Error: Dataset {dataset} not available!")

    return train_loader, test_loader, input_size, num_classes, num_channels


def load_MNIST(data_dir, split, batch_size, shuffle, kwargs):
    """Load and preprocess MNIST data, return data loader."""
    train_flag = True if split == "train" else False
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        data_dir, train=train_flag, download=True, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return data_loader


def load_SVHN(data_dir, split, batch_size, shuffle, kwargs):
    """Load and preprocess SVHN data, return data loader.
    Note: 73257 digits for training, 26032 digits for testing
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2862,), (0.3299,))]
    )
    dataset = datasets.SVHN(
        root=data_dir, split=split, transform=transform, download=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return data_loader


def load_svhn_resnet18(data_dir, split, batch_size, shuffle, kwargs):
    """Load SVHN preprocessed through resnet18 that was trained on CIFAR10, return data loader."""
    train_flag = True if split == "train" else False

    dataset = cu_datasets.svhn_resnet18(
        root="./data", train=train_flag, transform=None
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return data_loader


def load_CIFAR10(split, batch_size, shuffle, no_da, kwargs):
    """Load and preprocess CIFAR10 data, return data loader."""
    train_flag = True if split == "train" else False

    # transform from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    transform = {}
    if no_da:  # no data augment
        transform["train"] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform["train"] = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    transform["test"] = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=train_flag,
        download=True,
        transform=transform[split],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return data_loader


def load_cifar100_resnet50(data_dir, split, batch_size, shuffle, kwargs):
    """Load CIFAR100 preprocessed through resnet50, return data loader."""
    train_flag = True if split == "train" else False

    dataset = cu_datasets.cifar100_resnet50(
        root="./data", train=train_flag, transform=None
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return data_loader


def load_CIFAR100(split, batch_size, shuffle, img_size, kwargs):
    """Load and preprocess CIFAR100 data, return data loader."""
    train_flag = True if split == "train" else False

    transform = {}

    # transform['train'] = transforms.Compose([
    #                         transforms.RandomResizedCrop(img_size),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    #                     ])

    transform["train"] = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
            ),
        ]
    )

    transform["test"] = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
            ),
        ]
    )

    dataset = torchvision.datasets.CIFAR100(
        root="./data/cifar100",
        train=train_flag,
        download=True,
        transform=transform[split],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )

    return data_loader


def load_FashionMNIST(data_dir, split, batch_size, shuffle, kwargs):
    """Load and preprocess FashionMNIST data, return data loader."""
    train_flag = True if split == "train" else False
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2862,), (0.3299,))]
    )
    dataset = datasets.FashionMNIST(
        data_dir, train=train_flag, download=True, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return data_loader


def evaluate_TL(model, feature_extractor, data_loader, device, criterion):
    """Evaluate model, return prediction accuracy and loss."""
    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            labels = labels.to(device)
            inputs = inputs.to(device)

            # ==== pre-processing by resnet50
            inputs = feature_extractor(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            loss_sum += len(labels) * loss.item()
            correct += (predicted == labels).sum().item()
            total += len(labels)

    acc = correct / total
    loss = loss_sum / total

    return acc, loss


def evaluate(
    model, emb, data_loader, normalize_pixelwise, input_size, device, criterion
):
    """Evaluate model, return prediction accuracy and loss."""
    model.eval()
    if emb:
        emb.eval()

    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            labels = labels.to(device)

            if emb:
                inputs = inputs.to(device)
                inputs = emb(inputs).view(inputs.size(0), -1)
            else:
                num_channels = inputs.shape[1]
                if num_channels == 1:
                    inputs = inputs.reshape(-1, input_size).to(device)
                else:
                    inputs = inputs.to(device)
                if normalize_pixelwise:
                    inputs = pixelwise_normalization(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            loss_sum += len(labels) * loss.item()
            correct += (predicted == labels).sum().item()
            total += len(labels)

    acc = correct / total
    loss = loss_sum / total

    return acc, loss


def save_checkpoint(state, savepath):
    """Save model checkpoint."""
    # create (sub)dirs if not yet existent
    savedir = os.path.dirname(savepath)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    torch.save(state, f"{savepath}.ckpt")


###=== specifically for main_TL


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_dataset_TL(dataset, batch_size, test_batch_size, img_size, kwargs):

    if dataset == "CIFAR100":  # split, batch_size, shuffle, img_size, kwargs
        train_loader = load_CIFAR100(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            img_size=img_size,
            kwargs=kwargs,
        )
        test_loader = load_CIFAR100(
            split="test",
            batch_size=test_batch_size,
            shuffle=False,
            img_size=img_size,
            kwargs=kwargs,
        )
        num_channels = 3
        num_classes = 100
    else:
        print(f"Error: Dataset {dataset} not available!")

    return train_loader, test_loader, num_classes, num_channels
