
import torch.optim as optim


def training(net, data_loader, loss_function, epochs, device='cpu', training_loss_list=None):
    if training_loss_list is None:
        training_loss_list = []
    # Define optimizer
    optimizer = optim.Adam(net.parameters())

    n_images = len(data_loader.dataset)

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        # ----------------- TRAINING  -------------------- #
        net.train() # set model to training

        tot_error=0
        tot_images=0

        for batch_idx, data in enumerate(data_loader):
            inputs = data[0][0].to(device)
            t = data[0][1].to(device)
            labels = data[1].to(device)

            # Compute prediction (forward input in the model)
            outputs=net(inputs, t)

            # Compute prediction error with the loss function
            error=loss_function(outputs,labels)

            # Backpropagation
            net.zero_grad()
            error.backward()

            # Optimizer step
            optimizer.step()

            tot_error+=error*len(labels) # weighted average
            tot_images+=len(labels)

            print(f'Epoch {epoch+1}, loss: {tot_error/tot_images:.3g}', end = '\r')

            training_loss_list.append(error.detach().cpu().numpy())

        #TODO: improve the printing of the output
        print(f'Epoch {epoch+1}, loss: {tot_error/tot_images}')

    return training_loss_list
