import numpy as np
import torch
from torch.autograd import Variable
import os

def data_generator(folder, seq_length, overlap, phase = None, val_mask = np.arange(1,18,1, dtype=np.int), rng = None):
    # Reads all the experiment files in the given folder and returns numpy arrays with inputs and labels
    # The full experiment signal is broken into smaller time slices based on the supplied sampling frequency and desired "time_slice".
    # One can determine the allowed overlap (as a fraction) between slices. It may be beneficial to change everything to operate 
    # directly in absolute number of samples rather than the current time abstraction.

    if not (isinstance(phase, list) or phase==None):
        phase = [phase]

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    x_data = []
    y_data = []
    for path in files:
        raw_data = np.genfromtxt(os.path.join(folder,path),delimiter=',')

        for i in range(0,raw_data.shape[0]-seq_length,(seq_length-overlap)):
            if phase == None:
                x_data.append(raw_data[i:i+seq_length,val_mask])
                y_data.append(raw_data[i+seq_length-1,-1])
            elif all([p in phase for p in raw_data[i:i+seq_length,-2]]): 
                x_data.append(raw_data[i:i+seq_length,val_mask])
                y_data.append(raw_data[i+seq_length-1,-1])
    # This should be repeated for all files
    x_data, y_data = np.array(x_data,dtype=np.float32), np.array(y_data, dtype=np.int)
    if not rng == None:
            shuffle_seq = np.arange(x_data.shape[0])
            rng.shuffle(shuffle_seq)
            x_data = x_data[shuffle_seq,:,:]
            y_data = y_data[shuffle_seq]
    x_data,y_data = Variable(torch.from_numpy(x_data)), Variable(torch.from_numpy(y_data))
    #x_data = np.swapaxes(x_data,1,2) do this in main script instead
    return x_data, y_data



def data_generator_test(folder, seq_length, overlap, phase = None, val_mask = np.arange(1,18,1, dtype=np.int)):
    # Reads all the experiment files in the given folder and returns numpy arrays with inputs and labels
    # The full experiment signal is broken into smaller time slices based on the supplied sampling frequency and desired "time_slice".

    if not (isinstance(phase, list) or phase==None):
        phase = [phase]

    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    x_data = []
    y_data = []
    plotting_data = []
    for path in files:
        raw_data = np.genfromtxt(os.path.join(folder,path),delimiter=',')
        x_data_ = []
        y_data_ = []
        plotting_data_ = []
        for i in range(0,raw_data.shape[0]-seq_length,(seq_length-overlap)):
            if phase == None:
                x_data_.append(raw_data[i:i+seq_length,val_mask])
                y_data_.append(raw_data[i+seq_length-1,-1])
                plotting_data_.append(raw_data[i+seq_length-1,[0,11,12,-2]])
            elif all([p in phase for p in raw_data[i:i+seq_length,-2]]): 
                x_data_.append(raw_data[i:i+seq_length,val_mask])
                y_data_.append(raw_data[i+seq_length-1,-1])
                plotting_data_.append(raw_data[i+seq_length-1,[0,11,12,-2]])
        # This should be repeated for all files
        x_data_, y_data_ = np.array(x_data_,dtype=np.float32), np.array(y_data_, dtype=np.int)
        plotting_data_ = np.array(plotting_data_,dtype=np.float32)
        x_data_,y_data_ = Variable(torch.from_numpy(x_data_)), Variable(torch.from_numpy(y_data_))
        x_data.append(x_data_)
        y_data.append(y_data_)
        plotting_data.append(plotting_data_)
    #x_data = np.swapaxes(x_data,1,2) do this in main script instead
    return x_data, y_data, plotting_data,files
