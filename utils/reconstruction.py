import torch
import numpy as np

from utils.diffusionDataset import get_alpha_bar


def reconstruct_image_from_noise(noisy_image, noise, t, variance_schedule):
    """Reconstruct the image given the noisy image and the noise.

    Parameters
    ----------
    noisy_image : 3D numpy array
    noise : 3D numpy array
    t : int
        timestep.
    variance_schedule : array or array-like
        sequence of beta_t.

    Returns
    -------
    array
        original image
    """
    alpha_bar = get_alpha_bar(variance_schedule)[t]
    return (noisy_image - noise*np.sqrt(1-alpha_bar))/np.sqrt(alpha_bar)


def reconstruct_single_step(net, noisy_image, t, variance_schedule, device=None):
    """Reconstruct the image in a single step.

    Parameters
    ----------
    net : ``torch.nn.Module``
        the model that must reconstruct the image.
    noisy_image : 3D numpy array
            its shape must be ``(n_channels, spatial_1, spatial_2)``. Only a single image is
            processed, the batch dimension must be dropped.
    t : int
        timestep.
    variance_schedule : array or array-like
        sequence of beta_t.
    device : ``torch.device``, optional
        by default 'cpu'.

    Returns
    -------
    3D numpy array
        reconstructed image with shape ``(n_channels, spatial_1, spatial_2)``.
    """
    if device is None:
        device = torch.device('cpu')

    noisy_image_batch = torch.Tensor(noisy_image).reshape(1, *noisy_image.shape)
    t_batch = torch.Tensor([t])
    with torch.no_grad():
        noisy_image_device = noisy_image_batch.to(device)
        t_device = t_batch.to(device)
        pred_noise_device = net(noisy_image_device, t_device)

    pred_noise = pred_noise_device.detach().cpu().numpy()[0]
    rec_single_step = reconstruct_image_from_noise(noisy_image=noisy_image, noise=pred_noise, t=t, variance_schedule=variance_schedule)

    if 'cuda' in device.type:
        # free gpu memory
        del noisy_image_device
        del t_device
        del pred_noise_device
        torch.cuda.empty_cache()

    return rec_single_step


def reconstruct_sequentially(net, noisy_image, t, variance_schedule, device=None, sigma=None):
    """Sampling from Ho et. al (2020).

    Parameters
    ----------
    net : ``torch.nn.Module``
        the model that must reconstruct the image.
    noisy_image : 3D numpy array
            its shape must be ``(n_channels, spatial_1, spatial_2)``. Only a single image is
            processed, the batch dimension must be dropped.
    t : int
        timestep.
    variance_schedule : array or array-like
        sequence of beta_t.
    sigma : {int, array or array-like}, optional
        noise sequence to add at each reconstruction step.
    device : ``torch.device``, optional
        by default 'cpu'.

    Returns
    -------
    3D numpy array
        reconstructed image with shape ``(n_channels, spatial_1, spatial_2)``.
    """
    if device is None:
        device = torch.device('cpu')

    if sigma is None:
        sigma = np.zeros(t)

    with torch.no_grad():
        beta = torch.Tensor(variance_schedule).to(device)
        alpha_bar = torch.Tensor(get_alpha_bar(variance_schedule)).to(device)
        sigma = torch.Tensor(sigma).to(device)

        x = torch.Tensor(noisy_image.reshape(1, *noisy_image.shape)).to(device)

        print(f'T start = {t}')
        for ti in reversed(range(t)):
            print(f'Sampling: t={ti}'.ljust(50), end = '\r')
            t_tensor = torch.Tensor([ti]).to(device)
            pred_noise = net(x, t_tensor)
            x = (x - pred_noise*beta[ti]/torch.sqrt(1-alpha_bar[ti]))/torch.sqrt(1-beta[ti]) + sigma[ti]*torch.randn(x.shape).to(device)
        del t_tensor

        print('Sampling done.')
    return x.detach().cpu().numpy()[0]
