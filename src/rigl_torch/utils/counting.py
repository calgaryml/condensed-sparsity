# # https://github.com/google-research/google-research/blob/master/micronet_challenge/counting.py
# # https://github.com/google-research/rigl/blob/master/rigl/sparse_utils.py


# def get_stats(
#     masked_layers,
#     default_sparsity=0.8,
#     method="erdos_renyi",
#     custom_sparsities=None,
#     is_debug=False,
#     width=1.0,
#     first_layer_name="conv1",
#     last_layer_name="conv_preds",
#     param_size=32,
#     erk_power_scale=DEFAULT_ERK_SCALE,
# ):
#     """Given the Keras layer returns the size and FLOPS of the model.
#     Args:
#       masked_layers: list, of tf.keras.Layer.
#       default_sparsity: float, if 0 mask left intact, if greater than one, a
#         fraction of the ones in each mask is flipped to 0.
#       method: str, passed to the `.get_sparsities()` functions.
#       custom_sparsities: dictor None, sparsity of individual variables can be
#         overridden here. Key should point to the correct variable name, and
#         value should be in [0, 1].
#       is_debug: bool, if True prints individual stats for given layers.
#       width: float, multiplier for the individual layer widths.
#       first_layer_name: str, to scale the width correctly.
#       last_layer_name: str, to scale the width correctly.
#       param_size: int, number of bits to represent a single parameter.
#       erk_power_scale: float, passed to the get_sparsities function.
#     Returns:
#       total_flops, sum of multiply and add operations.
#       total_param_bits, total bits to represent the model during the inference.
#       real_sparsity, calculated independently omitting bias parameters.
#     """
#     if custom_sparsities is None:
#         custom_sparsities = {}
#     sparsities = get_sparsities(
#         [_get_kernel(l) for l in masked_layers],
#         method,
#         default_sparsity,
#         custom_sparsities,
#         lambda a: a,
#         erk_power_scale=erk_power_scale,
#     )
#     total_flops = 0
#     total_param_bits = 0
#     total_params = 0.0
#     n_zeros = 0.0
#     for layer in masked_layers:
#         kernel = _get_kernel(layer)
#         k_shape = kernel.shape.as_list()
#         d_in, d_out = 2, 3
#         # If fully connected change indices.
#         if len(k_shape) == 2:a
#             d_in, d_out = 0, 1
#         # and  k_shape[d_in] != 1 since depthwise
#         if not kernel.name.startswith(first_layer_name) and k_shape[d_in] != 1:
#             k_shape[d_in] = int(k_shape[d_in] * width)
#         if not kernel.name.startswith(last_layer_name) and k_shape[d_out] != 1:
#             k_shape[d_out] = int(k_shape[d_out] * width)
#         if is_debug:
#             print(
#                 kernel.name, layer.input_shape, k_shape, sparsities[kernel.name]
#             )

#         if isinstance(layer, tf.keras.layers.Conv2D):
#             layer_op = counting.Conv2D(
#                 layer.input_shape[1],
#                 k_shape,
#                 layer.strides,
#                 "same",
#                 True,
#                 "relu",
#             )
#         elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
#             layer_op = counting.DepthWiseConv2D(
#                 layer.input_shape[1],
#                 k_shape,
#                 layer.strides,
#                 "same",
#                 True,
#                 "relu",
#             )
#         elif isinstance(layer, tf.keras.layers.Dense):
#             layer_op = counting.FullyConnected(k_shape, True, "relu")
#         else:
#             raise ValueError("Should not happen.")
#         param_count, n_mults, n_adds = counting.count_ops(
#             layer_op, sparsities[kernel.name], param_size
#         )
#         total_param_bits += param_count
#         total_flops += n_mults + n_adds
#         n_param = np.prod(k_shape)
#         total_params += n_param
#         n_zeros += int(n_param * sparsities[kernel.name])

#     return total_flops, total_param_bits, n_zeros / total_params
