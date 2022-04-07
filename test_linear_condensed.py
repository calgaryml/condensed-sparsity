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
