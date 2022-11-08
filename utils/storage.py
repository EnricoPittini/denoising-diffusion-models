import os
import numpy as np

import torch
import torch.utils.data


def create_checkpoint_dict(net : torch.nn.Module,
                           epoch : int,
                           optimizer : torch.optim.Optimizer,
                           loss_history : list,
                           loss_history_val : list,
                           additional_info={}):
    """Get the training checkpoint dictionary.

    Parameters
    ----------
    net : torch.nn.Module
    epoch : int
    optimizer : torch.optim.Optimizer
    loss_history : list
    loss_history_val : list
    additional_info : dict, optional

    Returns
    -------
    dict
    """
    additional_info['model'] = str(type(net))
    additional_info['optimizer'] = str(type(optimizer))

    return {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'loss_history_val': loss_history_val,
            'additional_info': additional_info
           }


def save_checkpoint(checkpoint_dict, checkpoint_folder, clear_previous_checkpoints=True, keep_best=True, verbose=False):
    filename = 'checkpoint_' + f"{checkpoint_dict['epoch']}".zfill(4)

    # put best flag
    if checkpoint_dict['loss_history_val'][-1] == min(checkpoint_dict['loss_history_val']):
        filename += '_best'

    filename += '.ckpt'
    filepath = os.path.join(checkpoint_folder, filename)
    torch.save(checkpoint_dict, filepath)

    # save loss history train
    np.savetxt(os.path.join(checkpoint_folder, 'loss_history.csv'), checkpoint_dict['loss_history'], delimiter=',')
    # save loss history val
    np.savetxt(os.path.join(checkpoint_folder, 'loss_history_val.csv'), checkpoint_dict['loss_history'], delimiter=',')

    if verbose: print(f"Checkpoint saved: {filepath}.")

    if clear_previous_checkpoints:
        _clear_checkpoint_folder(checkpoint_folder, keep_best)
        if verbose: print('Cleared previous checkpoints.')


def _clear_checkpoint_folder(checkpoint_folder, keep_best):
    checkpoints = [i for i in os.listdir(checkpoint_folder) if i != 'loss_history.csv' and i != 'loss_history_val.csv']

    best_found = '_best' in checkpoints[-1]

    for c in reversed(checkpoints[:-1]):
        if best_found:
            os.remove(os.path.join(checkpoint_folder, c))
        else:
            if '_best' in c and keep_best:
                best_found = True
            else:
                os.remove(os.path.join(checkpoint_folder, c))


def load_checkpoint_dict(checkpoint_folder : str):
    """Load training status from a checkpoint.

    Parameters
    ----------
    checkpoint_folder : str
        folder containing the checkpoint file.
    net : torch.nn.Module
    optimizer : torch.optim.Optimizer

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        if ``checkpoint_folder`` does not exist.
    """
    if not os.path.exists(checkpoint_folder):
        raise FileNotFoundError(f"The folder {checkpoint_folder} does not exist.")

    if len(os.listdir(checkpoint_folder)) == 0:
        print(f"No checkpoint found in {checkpoint_folder}, using default initialization.")
        return None

    filename = [i for i in os.listdir(checkpoint_folder) if i != 'loss_history.csv' and i != 'loss_history_val.csv'][-1]
    filepath = os.path.join(checkpoint_folder, filename)

    print(f"Loading checkpoint: {filepath}")
    checkpoint_dict = torch.load(filepath)

    return checkpoint_dict


def load_checkpoint(checkpoint_folder : str,
                    net : torch.nn.Module,
                    optimizer : torch.optim.Optimizer):
    """Load training status from a checkpoint.

    Parameters
    ----------
    checkpoint_folder : str
        folder containing the checkpoint file.
    net : torch.nn.Module
    optimizer : torch.optim.Optimizer

    Returns
    -------
    epoch : int
    net : torch.nn.Module
        the model with the loaded ``state_dict``.
    optimizer : torch.optim.Optimizer
        the optimizer with the loaded ``state_dict``.
    loss_history : list
    loss_history_val : list
    additional_info : dict

    None if ``checkpoint_folder`` is empty.

    Raises
    ------
    FileNotFoundError
        if ``checkpoint_folder`` does not exist.
    """
    checkpoint = load_checkpoint_dict(checkpoint_folder)

    epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_history = checkpoint['loss_history']
    loss_history_val = checkpoint['loss_history_val']
    additional_info = checkpoint['additional_info']

    return epoch, net, optimizer, loss_history, loss_history_val, additional_info


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
