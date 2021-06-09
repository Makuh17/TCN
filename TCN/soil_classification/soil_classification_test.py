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
from TCN.soil_classification.model import TCN
from TCN.soil_classification.utils import data_generator
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Soil Classification')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
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


data_folder = "/home/mads/git/TCN/TCN/soil_classification/data/prelim_downsample"

#updated until here
#TODO:
    # run evaluation, get confusion matrix
    # find a proper way to do normalisation   

def train(epoch, X_train, Y_train):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    full_loss = 0
    idx = np.arange(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        rng.shuffle(idx)
        if i + batch_size > X_train.size(0):
            x, y = X_train[idx[i:]], Y_train[idx[i:]]
        else:
            x, y = X_train[idx[i:(i+batch_size)]], Y_train[idx[i:(i+batch_size)]]
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
        full_loss += loss.item()

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0
    full_loss = full_loss/(batch_idx-1)
    return full_loss
            
def evaluate(X_test, Y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.nll_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        pred = np.argmax(output.numpy(),axis=1)
        report = metrics.classification_report(Y_test, pred, output_dict = True)
        return test_loss.item(), report

def get_shuffled_train_test(X,Y, test_pct = 15, normalize = True, seed = 1111):
    shuffle_seq = np.arange(X.shape[0])
    rng.shuffle(shuffle_seq)
    X = X[shuffle_seq,:,:]
    Y = Y[shuffle_seq]
    X,Y = Variable(torch.from_numpy(X)), Variable(torch.from_numpy(Y))
    X_train, X_test = X[int(X.size()[0]/100*test_pct):], X[:int(X.size()[0]/100*test_pct)]
    Y_train, Y_test = Y[int(X.size()[0]/100*test_pct):], Y[:int(X.size()[0]/100*test_pct)]

    # Normalisation has to happen after division into train and test data, since it must only be based on training data.
    if normalize:
        mins, _ = torch.min(torch.min(X_train,1)[0],0)
        maxs, _ = torch.max(torch.max(X_train,1)[0],0)
        X_train, X_test = (X_train-mins)/(maxs-mins), (X_test-mins)/(maxs-mins)
    X_train, X_test = X_train.permute(0,2,1), X_test.permute(0,2,1)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    # Define tests
    sequence_length = [10,25]#[50, 100, 250, 500]
    # [All, IMU, Torques]
    #variable_set = [range(LIN_ACC_X,PITCH+1,1),list(range(POS_X, ANG+1)) + list(range(ORIENT_X,ORIENT_W+1)), range(ORIENT_X,ORIENT_W+1), range(LIN_ACC_X,ANG_VEL_Z+1,1),range(BOOM,PITCH+1,1)]
    #variable_set_name = ["all", "orient+pose","imu_no_orient", "torques"]
    variable_set = [range(LIN_ACC_X,ANG_VEL_Z+1,1),range(BOOM,PITCH+1,1)]
    variable_set_name = ["imu_no_orient", "torques"]
    # Include the slightly troublesome air-class?
    air_class = [True, False]
    for seq_len in sequence_length:
        for val_mask, val_mask_name in zip(variable_set, variable_set_name):
            for air in air_class:
                # for now we just use 50% overlap
                overlap = int(seq_len / 2)
                print(overlap)
                input_channels = len(val_mask)
                n_classes = 3
                print("Producing data...")
                #X_, Y = data_generator(data_folder, seq_len, overlap, val_mask=val_mask)
                train_folder = "/home/mads/git/TCN/TCN/soil_classification/data/prelim_downsample_train"
                test_folder = "/home/mads/git/TCN/TCN/soil_classification/data/prelim_downsample_test"
                X_train, Y_train = data_generator(train_folder, seq_len, overlap, val_mask=val_mask)
                X_test, Y_test = data_generator(test_folder, seq_len, overlap, val_mask=val_mask)
                if not air:
                    #remove_air = Y>0
                    #X, Y = X[remove_air], Y[remove_air]
                    remove_air = Y_train>0
                    X_train, Y_train = X_train[remove_air], Y_train[remove_air]
                    remove_air = Y_test>0
                    X_test, Y_test= X_test[remove_air], Y_test[remove_air]

                shuffle_seq = np.arange(X_train.shape[0])
                rng.shuffle(shuffle_seq)
                X_train = X_train[shuffle_seq,:,:]
                Y_train = Y_train[shuffle_seq]
                X_train,Y_train = Variable(torch.from_numpy(X_train)), Variable(torch.from_numpy(Y_train))
                X_test,Y_test = Variable(torch.from_numpy(X_test)), Variable(torch.from_numpy(Y_test))

                mins, _ = torch.min(torch.min(X_train,1)[0],0)
                maxs, _ = torch.max(torch.max(X_train,1)[0],0)
                X_train, X_test = (X_train-mins)/(maxs-mins), (X_test-mins)/(maxs-mins)
                X_train, X_test = X_train.permute(0,2,1), X_test.permute(0,2,1)

                print(X_train.size(), X_test.size())
#.....................................................................................................
                #X_train, Y_train, X_test, Y_test = get_shuffled_train_test(X,Y, test_pct=15, normalize=args.normalize, seed = args.seed)

                #Note on levels: n = ceil(log_2((seq_length-1)*(2-1)/((kernel_size-1)*2)+1)
                # from https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4
                # from quick calculations, 6 levels should be enough to cover 500 seq lenght 

                channel_sizes = [nhid]*levels
                kernel_size = ksize
                dropout = args.dropout
                model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

                optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)

                if args.cuda:
                    model.cuda()
                    X_train = X_train.cuda()
                    Y_train = Y_train.cuda()
                    X_test = X_test.cuda()
                    Y_test = Y_test.cuda()

                train_loss = []
                test_loss = []
                test_reports = []
                train_reports = []
                model_name = "seq"+str(seq_len)+"_"+val_mask_name+"_air"+str(air)
                model_folder = "models_3/"+model_name
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                for ep in range(1, epochs+1):
                    train_loss.append(train(ep, X_train, Y_train))
                    #train_losses.append(train_loss)
                    test_loss_, test_report = evaluate(X_test, Y_test)
                    _, train_report = evaluate(X_train, Y_train)
                    test_loss.append(test_loss_)
                    test_reports.append(test_report)
                    train_reports.append(train_report)
                    # Save
                    torch.save(model.state_dict(), model_folder+"/"+model_name+"_ep"+str(ep)+".pt")
                FORMAT = '%Y-%m-%-d_%H%M%S'
                stamp = datetime.now().strftime(FORMAT)
                l = []
                for key in list(test_report.keys())[:-3]:
                    for key2 in list(test_report.get(key).keys())[:-1]:
                        l.append(key+"_"+key2)
                l.append("accuracy")        
                with open('log/grid_test_10/train_test_'+model_name+stamp+'.csv', 'w') as f:
                    f.write(f"epoch, train_loss, test_loss, "+', '.join(l)+", train_accuracy"+"\n")
                    for loss in enumerate(zip(train_loss, test_loss, test_reports, train_reports)):
                        l = []
                        for key in list(loss[1][2].keys())[:-3]:
                            for key2 in list(loss[1][2].get(key).keys())[:-1]:
                                l.append(f"{loss[1][2].get(key).get(key2)}") 
                        l.append(f"{loss[1][2].get('accuracy')}")
                        f.write(f"{loss[0]+1},{loss[1][0]}, {loss[1][1]}, "+', '.join(l)+f",{loss[1][3].get('accuracy')}"+"\n")
