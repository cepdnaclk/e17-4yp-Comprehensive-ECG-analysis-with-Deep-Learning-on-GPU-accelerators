# def MAE(losses):
#     error_sum = 0
#     for loss in losses:
#         absolute_error = abs(loss - 0)  # Assuming 0 is the target value
#         error_sum += absolute_error

#     mean_absolute_error = error_sum / len(losses)
#     return mean_absolute_error


import numpy as np
import torch
import pandas as pd
import os

#from ECGDataSet import pr_max_val, pr_min_val, qt_max_val, qt_min_val, qrs_max_val, qrs_min_val


 
# def checkpoint(model, filename):
#     torch.save(model.state_dict(), filename)
    
# def resume(model, filename):
#     model.load_state_dict(torch.load(filename))

def train(dataloader, model, loss_fn, optimizer, device, epoch):
    size = len(dataloader.dataset)

    # cosineannealing warm restrats
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min = 0.001)
    model.train()
    

    train_losses_epoch = [] 
    for batch, (X, y) in enumerate(dataloader):
        #print(X.shape)
        #print(y.shape)
        #exit()
        X, y = X.to(device), y.to(device)

        #print(X.shape)
        # Compute prediction error
        pred = model(X)
        #print(X)
        #exit()
        #print(pred.shape)
        #print('y shape down')
        #print(y.shape)
        loss = loss_fn(pred, y)
        #print('pred down')
        #print(pred)
        #print(y)
        #print(loss)

        # zero out the gradients, perform the backpropagation step,
		# and update the weights
	    #optimizer.zero_grad()
        #loss.backward()
		#optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch / size)

        # Backpropagation
        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()
        
        train_losses_epoch.append(loss.item())
    
    return np.mean(train_losses_epoch)


def validate(dataloader, model, loss_fn, device, y_parameter):
    #print(pr_max_val)
    #print(pr_min_val)
    #print(qt_max_val)
    #print(qt_min_val)
    #print(qrs_max_val)
    #print(qrs_min_val)

    model.eval()  # Set the model to evaluation mode
    val_losses_epoch = []
    #val_real_epoch = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            #print(X.shape)
            #print(y.shape)
            #exit()
            # Compute predictions
            pred = model(X)
            loss = loss_fn(pred, y)

            #print(X)
            #print(y)        # y is inf?
            #print(pred)
            #print(loss)
            #exit()

            #convert pred to real values
            #if (y_parameter == 'hr'):
                #val_losses_epoch.append(loss.item())
                #val_real_epoch.append(loss.item())
            #elif (y_parameter == 'pr'):
                #predr = pred * (pr_max_val - pr_min_val) + pr_min_val
                #lossr = loss_fn(predr, y)
                #val_real_epoch.append(lossr.item())
            #elif (y_parameter == 'qt'):
                #predr = pred * (qt_max_val - qt_min_val) + qt_min_val
                #lossr = loss_fn(predr, y)
                #val_real_epoch.append(lossr.item())
            #elif (y_parameter == 'qrs'):
                #predr = pred * (qrs_max_val - qrs_min_val) + qrs_min_val
                #lossr = loss_fn(predr, y)
                #val_real_epoch.append(lossr.item())

            val_losses_epoch.append(loss.item())

    return np.mean(val_losses_epoch) #, np.mean(val_real_epoch)

def validate_notscaled(dataloader, model, loss_fn, device, y_parameter):

    # get the min max values from train dataset
    train = os.path.join(os.path.dirname(os.getcwd()), 'data', 'deepfake_ecg_full_train_validation_test/clean', 'train' + '.csv')
    train_df = pd.read_csv(train)

    if (y_parameter == 'pr'):
        column = train_df[y_parameter]
        pr_min_val = column.min()
        pr_max_val = column.max()
        
    elif (y_parameter == 'qt'):
        column = train_df[y_parameter]
        qt_min_val = column.min()
        qt_max_val = column.max()
        
    elif (y_parameter == 'qrs'):
        column = train_df[y_parameter]
        qrs_min_val = column.min()
        qrs_max_val = column.max()

    model.eval()  # Set the model to evaluation mode
    #val_losses_epoch = []
    val_real_epoch = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute predictions
            pred = model(X)
            loss = loss_fn(pred, y)

            #convert pred to real values
            if (y_parameter == 'hr'):
                #val_losses_epoch.append(loss.item())
                val_real_epoch.append(loss.item())
            elif (y_parameter == 'pr'):
                predr = pred * (pr_max_val - pr_min_val) + pr_min_val
                lossr = loss_fn(predr, y)
                val_real_epoch.append(lossr.item())
            elif (y_parameter == 'qt'):
                predr = pred * (qt_max_val - qt_min_val) + qt_min_val
                lossr = loss_fn(predr, y)
                val_real_epoch.append(lossr.item())
            elif (y_parameter == 'qrs'):
                predr = pred * (qrs_max_val - qrs_min_val) + qrs_min_val
                lossr = loss_fn(predr, y)
                val_real_epoch.append(lossr.item())

            #val_losses_epoch.append(loss.item())

    return  np.mean(val_real_epoch)

def validate_notscaled_tl(dataloader, model, loss_fn, device, y_parameter, pr_min_val, pr_max_val, qt_min_val, qt_max_val, qrs_min_val, qrs_max_val):

    model.eval()  # Set the model to evaluation mode
    #val_losses_epoch = []
    val_real_epoch = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute predictions
            pred = model(X)
            loss = loss_fn(pred, y)

            #convert pred to real values
            if (y_parameter == 'hr'):
                #val_losses_epoch.append(loss.item())
                val_real_epoch.append(loss.item())
            elif (y_parameter == 'pr'):
                predr = pred * (pr_max_val - pr_min_val) + pr_min_val
                lossr = loss_fn(predr, y)
                val_real_epoch.append(lossr.item())
                #print(pred)
                #print(y)
                #print(predr)
                #print(lossr)
                #exit()
            elif (y_parameter == 'qt'):
                predr = pred * (qt_max_val - qt_min_val) + qt_min_val
                lossr = loss_fn(predr, y)
                val_real_epoch.append(lossr.item())
            elif (y_parameter == 'qrs'):
                predr = pred * (qrs_max_val - qrs_min_val) + qrs_min_val
                lossr = loss_fn(predr, y)
                val_real_epoch.append(lossr.item())

            #val_losses_epoch.append(loss.item())

    return  np.mean(val_real_epoch)