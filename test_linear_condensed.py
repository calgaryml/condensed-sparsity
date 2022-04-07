import torch

from torch.nn import Linear
from condensed_sparse import LinearCondensed

def test_linear_condensed_output_size_type():
    """
    Test output shape/type.
    """
    batch_size = 2
    in_features = 784
    out_features = 8
    fan_in = 2

    test_input = torch.zeros(batch_size, in_features, dtype=torch.float32)

    condensed_layer = LinearCondensed(in_features=in_features, out_features=out_features, fan_in=fan_in, fan_out_const=False)

    test_output = condensed_layer(test_input)

    assert test_output.shape == (batch_size, out_features)
    assert test_output.dtype == torch.float32


# def test_spe_matmul():
#     """
#     Test output shape/type.
#     """
#     batch_size = 1
#     input_len = 1
#     num_units = 1
#     fan_in = 1

#     test_input = torch.zeros(batch_size, input_len, dtype=torch.float32)

#     condensed_layer = LinearCondensed(fan_in, num_units, bias=True)

#     dense_layer = Linear(fan_in, num_units, bias=True)

#     condensed_output = condensed_layer(test_input)
#     dense_output = dense_layer(test_input)

#     assert condensed_output.shape == (batch_size, num_units)
#     assert condensed_output.dtype == torch.float32

#     assert dense_output.shape == (batch_size, num_units)
#     assert dense_output.dtype == torch.float32

#     torch.testing.assert_close(dense_output, condensed_output)
