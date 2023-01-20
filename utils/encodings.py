import torch


def pos_encoding(t, channels, spatial_dimensions, device='cpu'):
    """Positional encoding of the timestep.
    Each channels is a component of the fourier series of t.

    Parameters
    ----------
    t : Tensor
        timestep. Its shape must be (1,).
    channels : int
        corresponds to the number of fourier components of the series.
        Half the channels are the sin components and half for the cos  components. Must be an even number.
    spatial_dimensions : tuple of int
    device : str, optional
        by default 'cpu'.

    Returns
    -------
    Tensor
        positional encoding of t with shape
        ``(batch_size, channels, spatial_dimensions[0], spatial_dimensions[1])``
    """
    #TODO: check that 10000!
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=device).float() / channels)
    )

    pos_enc_a = torch.sin(torch.outer(t, inv_freq))
    pos_enc_b = torch.cos(torch.outer(t, inv_freq))

    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
    return pos_enc.reshape(-1, channels, 1, 1).repeat(1, 1, spatial_dimensions[0], spatial_dimensions[1])
