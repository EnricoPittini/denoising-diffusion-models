import os
import time
import numpy as np

import torch
import torch.utils.data
import torch.optim as optim


def create_checkpoint_dict(net : torch.nn.Module,
                           epoch : int,
                           optimizer : torch.optim.Optimizer,
                           loss_history : list,
                           additional_info={}):
    """Get the training checkpoint dictionary.

    Parameters
    ----------
    net : torch.nn.Module
    epoch : int
    optimizer : torch.optim.Optimizer
    loss_history : list
    additional_info : dict, optional

    Returns
    -------
    dict
    """
    additional_info['model'] = type(net)
    additional_info['optimizer'] = type(optimizer)

    return {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'additional_info': additional_info
           }


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

    filename = os.listdir(checkpoint_folder)[-1]
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
    additional_info = checkpoint['additional_info']

    return epoch, net, optimizer, loss_history, additional_info


def _clear_checkpoint_folder(checkpoint_folder, keep_best):
    checkpoints = [i for i in os.listdir(checkpoint_folder) if i != 'loss_history.csv']

    best_found = '_best' in checkpoints[-1]

    for c in reversed(checkpoints[:-1]):
        if best_found:
            os.remove(os.path.join(checkpoint_folder, c))
        else:
            if '_best' in c and keep_best:
                best_found = True
            else:
                os.remove(os.path.join(checkpoint_folder, c))


def train_model(net : torch.nn.Module,
                data_loader : torch.utils.data.DataLoader,
                loss_function : torch.nn.Module,
                epochs : int,
                optimizer : torch.optim.Optimizer = None,
                device : torch.device = None,
                checkpoint_folder : str = None,
                additional_info : dict = {},
                checkpoint_step : int = 1,
                clear_previous_checkpoints=True,
                keep_best=True,
                verbose=False):
    """Training loop.

    Parameters
    ----------
    net : torch.nn.Module
        the model to train
    data_loader : torch.utils.data.DataLoader
    loss_function : torch.nn.Module
    epochs : int
    optimizer : torch.optim.Optimizer, optional
        by default Adam.
    device : torch.device, optional
        cpu or cuda, by default cpu.
    checkpoint_folder : str, optional
        folder into which to save the training checkpoints. If not provided, no checkpoints are
        created.
    additional_info : dict, optional
        additional info to save alongside the checkpoint information.
    checkpoint_step : int, optional
        every how many epochs a checkpoint is created, by default 1.
    clear_previous_checkpoints : bool, optional
        if set to True, the previous checkpoints are deleted. The epoch number is appended at the
        end of the filename provided by ``checkpoint_filename``. By default False.
    keep_best : bool, optional
        if set to True, keeps also the checkpoint with the best loss. Has an effect only if
        ``clear_previous_checkpoints`` is set to True.
    verbose : bool, optional
        if true, prints each time chekpoints are created.

    Returns
    -------
    dict
        the checkpoint dictionary.
    """

    # -------------------- SETUP -------------------- #
    if device is None:
        device = torch.device('cpu')
    net.to(device)
    print(f"Device: {device}")

    if optimizer is None:
        optimizer = optim.Adam(net.parameters())

    save_checkpoints = checkpoint_folder is not None

    starting_epoch = 0
    loss_history = []

    scaler=torch.cuda.amp.GradScaler()

    # resume from previous checkpoint
    if checkpoint_folder is not None:
        if os.path.exists(checkpoint_folder):
            checkpoint = load_checkpoint(checkpoint_folder=checkpoint_folder, net=net, optimizer=optimizer)
            if checkpoint is not None:
                starting_epoch, net, optimizer, loss_history, additional_info = checkpoint
                print("Checkpoint loaded.")
        else:
            os.makedirs(checkpoint_folder)
            print(f"Created checkpoint folder {checkpoint_folder}")

    if not verbose: print(" ")

    # -------------------- TRAINING -------------------- #
    net.train()         # set model to training mode
    # loop for every epoch (training + evaluation)
    for i, epoch in enumerate(range(starting_epoch, epochs+starting_epoch)):
        if verbose: print(" ")
        print(f'Epoch: {epoch+1}/{epochs+starting_epoch}')

        tot_error=0
        tot_images=0
        start_time = time.time()

        for batch_idx, data in enumerate(data_loader):
            inputs = data[0][0].to(device)
            t = data[0][1].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                # Compute prediction (forward input in the model)
                outputs = net(inputs, t)

                # Compute prediction error with the loss function
                error = loss_function(outputs, labels)

            # Backpropagation
            #net.zero_grad()
            #error.backward()
            scaler.scale(error).backward()

            # Optimizer step
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            tot_error += error*len(labels)      # weighted average
            tot_images += len(labels)

            mse_loss = tot_error/tot_images

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)

            print(f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, loss: {mse_loss:.3g}'.ljust(80), end = '\r')

        print(f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, loss: {mse_loss:.3g}'.ljust(80))
        mse_loss_np = (mse_loss).detach().cpu().numpy()
        loss_history.append(mse_loss_np)

        # create checkpoint dictionary
        if i%checkpoint_step == 0:
            checkpoint_dict = create_checkpoint_dict(net=net,
                                                     epoch=epoch+1,
                                                     optimizer=optimizer,
                                                     loss_history=loss_history,
                                                     additional_info=additional_info)

            # save checkpoint dict if filename is provided
            if save_checkpoints:
                filename = 'checkpoint_' + f"{epoch+1}".zfill(4)

                # put best flag
                if mse_loss_np == min(loss_history):
                    filename += '_best'

                filename += '.ckpt'
                filepath = os.path.join(checkpoint_folder, filename)
                torch.save(checkpoint_dict, filepath)

                # save loss histoyy
                np.savetxt(os.path.join(checkpoint_folder, 'loss_history.csv'), loss_history, delimiter=',')

                if verbose: print(f"Checkpoint saved: {filepath}.")

                if clear_previous_checkpoints:
                    _clear_checkpoint_folder(checkpoint_folder, keep_best)
                    if verbose: print('Cleared previous checkpoints.')

    print('\nTraining done.')

    return checkpoint_dict
