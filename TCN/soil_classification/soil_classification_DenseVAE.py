from datetime import datetime
from pathlib import Path
from sklearn import metrics
import numpy as np
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.soil_classification.model import SimpleVAE
from TCN.soil_classification.utils import data_generator, data_generator_test
from TCN.soil_classification.model_VAE import DenseVAE
from torch.autograd import Variable
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Soil Classification')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--normalize', action='store_false',
                    help='min-max normalize inputs (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=16,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=250,
                    help='sequence length in seconds (default: 0.5)')
parser.add_argument('--samp_freq', type=int, default=500,
                    help='sampling frequency (default: 500)')
parser.add_argument('--overlap', type=int, default=0,
                    help='overlap for sequences (default: 0)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
rng = np.random.default_rng(args.seed)

# Definitions for data indicies
LIN_ACC_X = 1
LIN_ACC_Y = 2
LIN_ACC_Z = 3
ANG_VEL_X = 4
ANG_VEL_Y = 5
ANG_VEL_Z = 6
ORIENT_X = 7
ORIENT_Y = 8
ORIENT_Z = 9
ORIENT_W = 10
POS_X = 11
POS_Z = 12
ANG = 13
BOOM = 14
DIPPER = 15
TELE = 16
PITCH = 17

print(args)

batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
overlap = args.overlap
nhid = args.nhid
levels = args.levels
ksize = args.ksize
lr = args.lr
clip = args.clip
log_interval = args.log_interval

def train(epoch, X_train):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    full_loss = 0
    num_train_img = X_train.size(0)
    idx = np.arange(num_train_img)
    for i in range(0, X_train.size(0), batch_size):
        rng.shuffle(idx)
        if i + batch_size > X_train.size(0):
            x = X_train[idx[i:]]
        else:
            x = X_train[idx[i:(i+batch_size)]]
        bs = x.shape[0]
        optimizer.zero_grad()
        output = model(x)
        loss = model.loss_function(*output, M_N = 0.001)
        loss['loss'].backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss['loss'].item()
        full_loss += loss['loss'].item()

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0
    full_loss = full_loss/(batch_idx-1)
    print('\n Train set: Average loss: {:.6f}\n'.format(full_loss))
    return full_loss

def evaluate(epoch, X_test):
    model.eval()
    with torch.no_grad():
        outputs = []
        test_losses = []
        for x in X_test:
            output= model(x)
            outputs.append(output)
        
        #regular output
        output_cat = [torch.cat([o[0]for o in outputs]),torch.cat([o[1]for o in outputs]),torch.cat([o[2]for o in outputs])]
        test_loss = model.loss_function(*output, M_N = 0.001)
        print('\n Test set: Average loss: {:.6f}\n'.format(test_loss['loss'].item()))
        return test_loss['loss'].item()

if __name__ == "__main__":
    # Define tests
    sequence_length = [128]#[50, 100, 250, 500]
    variable_set = [list(range(LIN_ACC_X,ANG_VEL_Z+1,1))+list(range(BOOM,PITCH+1,1))]
    variable_set_name = ["imu_no_orient+torques"]
    phase_set = [2, 1]

    train_folder = "/home/mads/git/TCN/TCN/soil_classification/data/exp_1604/train"
    test_folder = "/home/mads/git/TCN/TCN/soil_classification/data/exp_1604/test"
    
    seq_len = sequence_length[0]
    val_mask_name = variable_set_name[0]
    phase = phase_set[0]
    
    X_train, Y_train = data_generator(train_folder, seq_len, seq_len-1, phase =phase, val_mask=variable_set[0], rng=rng)
    #X_test, Y_test = data_generator(test_folder, seq_len, seq_len-1, phase =phase, val_mask=variable_set[0])
    X_test, Y_test, plotting, files = data_generator_test(test_folder, seq_len, seq_len-1, phase =phase, val_mask=variable_set[0])
    
    #if "seqnormed" in variable_set_name[0]: # we use the sequence normalization
     #   mins = torch.min(X_train,1)[0].unsqueeze(1)
      #  maxs = torch.max(X_train,1)[0].unsqueeze(1)
    #else:
     #   mins, _ = torch.min(torch.min(X_train,1)[0],0)
      #  maxs, _ = torch.max(torch.max(X_train,1)[0],0)

    m = X_train.mean(0, keepdim=True)
    s = X_train.std(0, unbiased=False, keepdim=True)
    X_train -= m
    X_train /= s

    #X_train = (X_train-mins)/(maxs-mins)
    X_train = X_train.permute(0,2,1)

    for traj_idx in range(len(X_test)):
       # if "_norm" in val_mask_name:
        #    X_test[traj_idx][:,:,0] = torch.linalg.norm(X_test[traj_idx][:,:,0:3], dim = 2)
         #   X_test[traj_idx][:,:,1] = torch.linalg.norm(X_test[traj_idx][:,:,3:6], dim = 2)
          #  X_test[traj_idx] = X_test[traj_idx][:,:,[0,1]+list(range(6,X_test[traj_idx].shape[2],1))]

        #if "seqnormed" in val_mask_name: # we use the sequence normalization
         #   mins = torch.min(X_test[traj_idx],1)[0].unsqueeze(1)
          #  maxs = torch.max(X_test[traj_idx],1)[0].unsqueeze(1)
           # X_test[traj_idx] = (X_test[traj_idx]-mins)/(maxs-mins)
        #else:
         #   X_test[traj_idx] = (X_test[traj_idx]-mins)/(maxs-mins)
        X_test[traj_idx] -= m
        X_test[traj_idx] /= s
        X_test[traj_idx] = X_test[traj_idx].permute(0,2,1)
    
    print(X_train.shape)
    input_channels = X_train.shape[1]

    channel_sizes = [nhid]*levels #[30,20,10,1]#
    kernel_size = ksize
    dropout = args.dropout
    latent_dim = 240
    scale = 10

    hidden_dims = [1000,800,500, 240]
    #model = TCNAE(input_channels, 10, 10, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    model = DenseVAE(input_channels, seq_len, latent_dim, hidden_dims)

    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)   

    if args.cuda:
        model.cuda()
        X_train = X_train.cuda()

    train_loss = []
    test_loss = []
    model_name = "DenseVAE_bs"+str(batch_size)+"_do"+str(int(dropout*100))+"_seq"+str(seq_len)+"_"+val_mask_name+"_phase"+str(phase)+"_nlf"+str(latent_dim)
    model_folder = "AE_model/"+model_name
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    FORMAT = '%Y-%m-%-d_%H%M%S'
    stamp = datetime.now().strftime(FORMAT)
    for ep in range(1, epochs+1):
        train_loss_ = train(ep, X_train)
        train_loss.append(train_loss_)
        test_loss_ = evaluate(ep, X_test)
        test_loss.append(test_loss_)
        torch.save(model.state_dict(), model_folder+"/"+model_name+"_ep"+str(ep)+".pt")
    with open('log/VAE/train_test_'+model_name+stamp+'.csv', 'w') as f:
        f.write(f"epoch, train_loss, test_loss \n")
        for ep, [tr_loss, te_loss] in enumerate(zip(train_loss,test_loss)):
            f.write(f"{ep},{tr_loss},{te_loss} \n")