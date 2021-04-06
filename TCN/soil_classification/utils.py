import numpy as np
import torch
from torch.autograd import Variable
import os

def data_generator(folder, seq_length, overlap, val_mask = np.arange(1,18,1, dtype=np.int)):
    # Reads all the experiment files in the given folder and returns numpy arrays with inputs and labels
    # The full experiment signal is broken into smaller time slices based on the supplied sampling frequency and desired "time_slice".
    # One can determine the allowed overlap (as a fraction) between slices. It may be beneficial to change everything to operate 
    # directly in absolute number of samples rather than the current time abstraction.


    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    x_data = []
    y_data = []
    for path in files:
        raw_data = np.genfromtxt(os.path.join(folder,path),delimiter=',')

        for i in range(0,raw_data.shape[0]-seq_length,(seq_length-overlap)):
            x_data.append(raw_data[i:i+seq_length,val_mask])
            y_data.append(raw_data[i+seq_length-1,-1])
    # This should be repeated for all files
    x_data, y_data = np.array(x_data,dtype=np.float32), np.array(y_data, dtype=np.int)
    #x_data = np.swapaxes(x_data,1,2) do this in main script instead
    return x_data, y_data
