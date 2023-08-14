# def MAE(losses):
#     error_sum = 0
#     for loss in losses:
#         absolute_error = abs(loss - 0)  # Assuming 0 is the target value
#         error_sum += absolute_error

#     mean_absolute_error = error_sum / len(losses)
#     return mean_absolute_error


import torch
 
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))