import torch


def pos_encoding(t, channels, spatial_dimensions, device='cpu'):
    """Positional encoding of the timestep. For a more detailed explanation:
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    We take the first `channels` components of the positional encoding.

    The integer timestep is transformed into a tensor of size `(channels, spatial_dimensions)`.
    The first half of channels contains the sine components, and the second half the cosine components.
    The values of the positional encoding are constant along the spatial dimensions and change along the channels.

    Parameters
    ----------
    t : Tensor
        timestep. Its shape must be `(batch_size, 1)`.
    channels : int
        corresponds to the number of components of the series.
        Half the channels are the sin components and half for the cos components. Must be an even number.
    spatial_dimensions : tuple of int
    device : str, optional
        by default 'cpu'.

    Returns
    -------
    Tensor
        positional encoding of t with shape
        ``(batch_size, channels, spatial_dimensions[0], spatial_dimensions[1])``
    """
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=device).float() / channels)
    )

    pos_enc_a = torch.sin(torch.outer(t, inv_freq))
    pos_enc_b = torch.cos(torch.outer(t, inv_freq))

    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
    return pos_enc.reshape(-1, channels, 1, 1).repeat(1, 1, spatial_dimensions[0], spatial_dimensions[1])
