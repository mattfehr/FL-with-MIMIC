import torch

def train_model(model,device, optimizer ,data_loader, loss_module, num_epochs=100):
    best_model = None
    best_loss  = 1

    model.train()
    train_losses  = []
    epoch_losses  = []
    for epoch in (range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:
            # Put training data into GPU
            data_inputs = data_inputs.to(device, non_blocking=True)
            data_labels = data_labels.type(torch.float32).to(device, non_blocking=True)

            preds, alpha = model(data_inputs)
            preds = preds.type(torch.float32).to(device)
            
            loss  = loss_module(preds, data_labels)
            # Save the model
            if best_loss > loss.item():
                best_loss  = loss.item()
                best_model = model.state_dict()
                
            epoch_loss += loss.item()
            train_losses.append(loss.item())

            # Zero the gradients
            optimizer.zero_grad()
            loss     .backward()
            optimizer.step()
        # Plot the average loss per epoch
        epoch_loss /= len(data_loader)
        epoch_losses.append(epoch_loss)
    return train_losses, best_model, best_loss