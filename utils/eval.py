import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.reconstruction import reconstruct_sequentially, reconstruct_single_step


def _visualize_images(original_image, noisy_image, rec_sequential, rec_single_step, t=None):
    if t is not None:
        print(f't = {t}')
    print(f'Noise variance = {np.std(original_image - noisy_image)**2:.3g}')
    print(f'MSE one step = {np.mean((rec_single_step - original_image)**2):.3g}')
    print(f'MSE sequential = {np.mean((rec_sequential - original_image)**2):.3g}')

    w, h, dpi = 800, 800, 100
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(w/dpi, h/dpi), dpi=dpi)
    axs = axs.ravel()

    axs[0].imshow(original_image.clip(0, 1).transpose(1, 2, 0))
    axs[0].set_title('Original image')
    axs[1].imshow(noisy_image.clip(0, 1).transpose(1, 2, 0))
    axs[1].set_title('Noisy input image')
    axs[2].imshow(rec_single_step.clip(0, 1).transpose(1, 2, 0))
    axs[2].set_title('Reconstructed image (1 step)')
    axs[3].imshow(rec_sequential.clip(0, 1).transpose(1, 2, 0))
    axs[3].set_title('Reconstructed image (sequential)')

    for ax in axs:
        ax.set_axis_off()

    fig.tight_layout()
    plt.show()


def visualize_single_reconstruction(net, original_image, noisy_image, t, variance_schedule, device=None, sigma=None):
    """Visualize a single reconstruction of an image with sequential and single step reconstruction.

    Parameters
    ----------
    net : torch.nn.Module
        the model that must reconstruct the image.
    original_image : 3D numpy array
        The original image to be reconstructed. Its shape must be ``(3, height, width)``.
    noisy_image : 3D numpy array
        The noisy image. Its shape must be ``(3, height, width)``.
    t : int
        timestep
    variance_schedule : array or array-like
        sequence for beta_t
    device : string, optioal
    """
    rec_sequential = reconstruct_sequentially(net, noisy_image, t, variance_schedule, device, sigma)
    rec_single_step = reconstruct_single_step(net, noisy_image, t, variance_schedule, device)

    _visualize_images(original_image, noisy_image, rec_sequential, rec_single_step, t)
