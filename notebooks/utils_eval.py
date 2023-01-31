def get_num_allzero_kernels(w):
    "Counts the number of all-zero kernels in a given conv weight tensor."
    num_filters = w.shape[0]
    # zero_kernel_inds=[]
    count = 0
    for filter_nr in range(num_filters):
        for k_ind, kernel in enumerate(w[filter_nr]):
            if not kernel.any():
                # zero_kernel_inds.append((filter_nr,k_ind))
                count += 1
    return count


def get_num_kernels(w):
    "Counts the number of kernels in a given conv weight tensor."
    num_filters = w.shape[0]
    num_kernels_per_filter = w.shape[1]
    return num_filters * num_kernels_per_filter


def get_num_allzero_filters(w):
    """Counts the number of all-zero filters in a given conv weight tensor,
    which corresponds to the number of ablated neurons
    (neurons = output channels = filters).
    """
    num_out_channels = w.shape[0]
    count = 0
    for n in range(num_out_channels):
        filter = w[n]
        if not filter.any():
            count += 1
    return count


def get_num_allzero_fanout(w):
    """Counts the number of all-zero fan-out in a given conv weight tensor."""
    num_in_channels = w.shape[1]

    count = 0
    for depth_ind in range(num_in_channels):
        # tensor.any() = True if any one of the elements in tensor is nonzero
        if not w[:, depth_ind, :, :].any():
            count += 1
    return count
