import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.reconstruction import reconstruct_sequentially, reconstruct_single_step, reconstruct_step


def load_weights(net : torch.nn.Module, checkpoint_filename : str):
    """Load weights from a checkpoint. The model is automatically set to eval mode.

    Parameters
    ----------
    net : torch.nn.Module
    checkpoint_filename : str
    """
    if not os.path.exists(checkpoint_filename):
        raise FileNotFoundError(f"The file {checkpoint_filename} does not exist.")

    checkpoint = torch.load(checkpoint_filename)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()


def _visualize_images(original_image,
                      noisy_image,
                      rec_baseline,
                      rec_single_step,
                      rec_sequential,
                      rec_step,
                      step,
                      t=None):
    if t is not None:
        print(f't = {t}')
    print(f'Noise variance = {np.std(original_image - noisy_image)**2:.3g}')
    print(f'MSE baseline bilateral = {np.mean((rec_baseline - original_image)**2):.3g}')
    print(f'MSE one step = {np.mean((rec_single_step - original_image)**2):.3g}')
    print(f'MSE sequential = {np.mean((rec_sequential - original_image)**2):.3g}')
    print(f'MSE step sequential = {np.mean((rec_step - original_image)**2):.3g}')

    nrows = int(2 + np.ceil((len(rec_step)-1)/3))
    ncols = 3
    w, h, dpi = 1200, int(1200*nrows/ncols), 100
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(w/dpi, h/dpi), dpi=dpi)

    axs = axs.ravel()

    axs[0].imshow(original_image.transpose(1, 2, 0))
    axs[0].set_title('Original image')

    axs[1].imshow(noisy_image.transpose(1, 2, 0))
    axs[1].set_title('Noisy input image')

    axs[2].imshow(rec_baseline.transpose(1, 2, 0))
    axs[2].set_title('Bilateral filter')

    axs[3].imshow(rec_single_step.transpose(1, 2, 0))
    axs[3].set_title('single step')

    axs[4].imshow(rec_sequential.transpose(1, 2, 0))
    axs[4].set_title('sequential')

    for i, im in enumerate(rec_step):
        axs[i+5].imshow(im.transpose(1, 2, 0))
        axs[i+5].set_title(f'step = {step[i]}')

    for ax in axs:
        ax.set_axis_off()

    fig.tight_layout()
    plt.show()


def visualize_single_reconstruction(net,
                                    original_image,
                                    noisy_image,
                                    t,
                                    variance_schedule,
                                    device=None,
                                    sigma=None,
                                    step=1,
                                    output_folder=None,
                                    bilateral_parameters=(2, 0.7)):
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
    device : string, optional
    sigma : {int, array or array-like}, optional
        noise sequence to add at each sequential reconstruction step.
    step : int, optional
        skip step in step reconstruction.
    output_folder : str, optional
        if provided, save the original image, noisy image and reconstructinos output_folder.
    bilateral_parameter : tuple(float, float), optional
        set the parameters for the baseline bilateral filtering. The elements are
        ``(sigmaSpace, sigmaColor)``. Default is (2, 0.7).
    """

    if not hasattr(step, "__getitem__"):
        step = [step]

    rec_baseline = cv2.bilateralFilter(np.asarray(noisy_image, dtype=np.float32).transpose(1, 2, 0),
                                                  d=-1,
                                                  sigmaSpace=bilateral_parameters[0],
                                                  sigmaColor=bilateral_parameters[1])

    rec_sequential = reconstruct_sequentially(net, noisy_image, t, variance_schedule, device, sigma)
    rec_single_step = reconstruct_single_step(net, noisy_image, t, variance_schedule, device)

    rec_step = []
    for s in step:
        rec_step.append(reconstruct_step(net, noisy_image, t, s, variance_schedule, device))

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.imsave(os.path.join(output_folder, 'original.png'), original_image.transpose(1, 2, 0))
        print('original saved')
        plt.imsave(os.path.join(output_folder, 'noisy.png'), noisy_image.transpose(1, 2, 0))
        print('noisy saved')
        plt.imsave(os.path.join(output_folder, 'baseline.png'), rec_baseline)
        print('baseline saved')
        plt.imsave(os.path.join(output_folder, 'rec_sequential.png'), rec_sequential.transpose(1, 2, 0))
        print('seq saved')
        plt.imsave(os.path.join(output_folder, 'rec_single_step.png'), rec_single_step.transpose(1, 2, 0))
        print('single step saved')

        for s, im in zip(step, rec_step):
            plt.imsave(os.path.join(output_folder, f'rec_step_{s}.png'), im.transpose(1, 2, 0))
            print(f'step {s} saved')

    _visualize_images(original_image=original_image,
                      noisy_image=noisy_image,
                      rec_baseline=rec_baseline.transpose(2, 0, 1),
                      rec_single_step=rec_single_step,
                      rec_sequential=rec_sequential,
                      rec_step=rec_step,
                      step=step,
                      t=t)
