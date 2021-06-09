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
from TCN.soil_classification.utils import data_generator, data_generator_test
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


data_folder = "/home/mads/git/TCN/TCN/soil_classification/data/prelim_downsample"

#updated until here
#TODO:
    # run evaluation, get confusion matrix
    # find a proper way to do normalisation   

def train(epoch, X_train, Y_train, augment=False):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    full_loss = 0
    idx = np.arange(X_train.size(0))
    preds = []
    truths = []
    for i in range(0, X_train.size(0), batch_size):
        rng.shuffle(idx)
        if i + batch_size > X_train.size(0):
            x, y = X_train[idx[i:]].clone(), Y_train[idx[i:]].clone()
        else:
            x, y = X_train[idx[i:(i+batch_size)]].clone(), Y_train[idx[i:(i+batch_size)]].clone()
        optimizer.zero_grad()
        if augment:
            x = rotation_augmentation(x)
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
        full_loss += loss.item()
        #for logging
        preds.append(output.detach())
        truths.append(y)

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0
    full_loss = full_loss/(batch_idx-1)
    truths = torch.cat(truths)
    preds = torch.cat(preds)
    print('\n Train set: Average loss: {:.6f}\n'.format(full_loss))
    preds = np.argmax(preds.numpy(),axis=1)
    report = metrics.classification_report(truths, preds, output_dict = True)
    return full_loss, report
            
def evaluate(X_test, Y_test, scenario = "Test"):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.nll_loss(output, Y_test)
        print('\n' + scenario + ' set: Average loss: {:.6f}\n'.format(test_loss.item()))
        pred = np.argmax(output.numpy(),axis=1)
        report = metrics.classification_report(Y_test, pred, output_dict = True)
        return test_loss.item(), report

def evaluate_test(X_test, Y_test, plot_data, files, name):
    model.eval()
    with torch.no_grad():
        output = []
        for x in X_test:
            output.append(model(x))

        rows = int((len(plotting)+1)/2)
        colors = ["r","g","b"]
        yticks = ['Air','Gravel','Dirt','Rocks']
        y_nums = np.array([0,1,2,3])
        phases = ["Penetration", "Scraping", "Lifting"]

        fig, ax1 = plt.subplots(rows,2,figsize = [16,4*rows])
        ax2 = np.empty_like(ax1)
        for a1,a2,d,f,y,out in zip(ax1.reshape(-1),ax2.reshape(-1),plot_data,files, Y_test,output):
            a2 = a1.twinx()
            a1.title.set_text(f)
            a1.set_ylim([-1, 1])
            a2.set_ylim([-1.3, 4.3])
            plt.yticks(y_nums,yticks)
            a1.axes.get_yaxis().set_ticks([])
            a1.set_xlabel("Time [s]")
            a1.set_ylabel("End-effector height")
            
            out = np.argmax(out.numpy(),axis=1)

            for phase in np.unique(d[:,-1]):
                a1.plot(d[d[:,-1]==phase,0],d[d[:,-1]==phase,2],color=colors[int(phase)-1],
                            linewidth=15, alpha=0.2, label = phases[int(phase)-1], solid_capstyle='butt')
                
            a2.plot(d[:,0],y, linewidth=9, label="True Class")
            a2.plot(d[:,0],out, linewidth=3, label="Predicted Class")
            a1.legend(loc=1)
            a2.legend(loc=4)
        fig.tight_layout(pad=2.0)
        plt.savefig(name+".pdf", dpi=150)
        plt.close()

        # -----------------------------------------------------------------------------------
        # Regular output.
        Y_test = torch.cat(Y_test)
        output = torch.cat(output)
        test_loss = F.nll_loss(output, Y_test)
        print('\n Test set: Average loss: {:.6f}\n'.format(test_loss.item()))
        pred = np.argmax(output.numpy(),axis=1)
        report = metrics.classification_report(Y_test, pred, output_dict = True)
        return test_loss.item(), report

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = rng.uniform(size=(3,))    
    theta, phi, z = randnums 
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )   
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def rotation_augmentation(x):
    x = x.clone()
    # angle = np.pi#rng.uniform(0,np.pi*2)
    # rotmat = torch.from_numpy(np.array([[1,0,0],
    #                                     [0, np.cos(angle), -np.sin(angle)], 
    #                                     [0, np.sin(angle), np.cos(angle)]], dtype=np.float64))
    rotmat = torch.from_numpy(rand_rotation_matrix())
    #print(x[i,0:3].shape)
    x[:,0:3] = torch.matmul(rotmat.float(),x[:,0:3])
    x[:,3:6] = torch.matmul(rotmat.float(),x[:,3:6])
    return x



if __name__ == "__main__":
    # Define tests
    sequence_length = [128]#[50, 100, 250, 500]
    # [All, IMU, Torques]
    #variable_set = [range(LIN_ACC_X,PITCH+1,1),list(range(POS_X, ANG+1)) + list(range(ORIENT_X,ORIENT_W+1)), range(ORIENT_X,ORIENT_W+1), range(LIN_ACC_X,ANG_VEL_Z+1,1),range(BOOM,PITCH+1,1)]
    #variable_set_name = ["all", "orient+pose","imu_no_orient", "torques"]
    #variable_set = [range(LIN_ACC_X,ANG_VEL_Z+1,1),range(BOOM,PITCH+1,1)]
    #variable_set_name = ["imu_no_orient", "torques"]
    variable_set = [list(range(LIN_ACC_X,ANG_VEL_Z+1,1))+list(range(BOOM,PITCH+1,1))]
    # variable_set_name = ["imu_no_orient+torques"]
    #variable_set = [list(range(ORIENT_X,ORIENT_W+1,1))]
    #variable_set_name = ["orient_only"]
    #variable_set = [list(range(LIN_ACC_X,ANG_VEL_Z+1,1))] 

    #variable_set = [list(range(LIN_ACC_X,LIN_ACC_Z+1,1))] 
    variable_set_name = ["imu_no_orient_norm_derivative+torque"] # "imu_no_orient","imu_no_orient_derivative",

    #variable_set = [list(range(LIN_ACC_X,ANG_VEL_Z+1,1))+list(range(BOOM,PITCH+1,1)), 
     #               list(range(LIN_ACC_X,ANG_VEL_Z+1,1)), 
      #              list(range(LIN_ACC_X,ANG_VEL_Z+1,1))+list(range(BOOM,PITCH+1,1)),
       #             list(range(LIN_ACC_X,ANG_VEL_Z+1,1))+list(range(BOOM,PITCH+1,1))]
    #variable_set_name = ["imu_no_orient+torques","imu_no_orient", "imu_no_orient_seqnormed+torques", "imu_no_orient_norm+torques"]
    
    #variable_set = [range(BOOM,PITCH+1,1)]
    #variable_set_name = ["torques"]
    phase_set = [2]
    seeds = [1] #[1,2,23,56,12,32,77,89,90,53]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        for seq_len in sequence_length:
            for val_mask, val_mask_name in zip(variable_set, variable_set_name):
                for phase in phase_set:
                    # maximum number of training data
                    overlap = seq_len - 1
                    print(overlap)
                    n_classes = 4
                    print("Producing data...")
                    #X_, Y = data_generator(data_folder, seq_len, overlap, val_mask=val_mask)
                    train_folder = "/home/mads/git/TCN/TCN/soil_classification/data/exp_1604/train"
                    test_folder = "/home/mads/git/TCN/TCN/soil_classification/data/exp_1604/test"
                
                    X_train, Y_train = data_generator(train_folder, seq_len, seq_len-1, phase =phase, val_mask=variable_set[0], rng=rng)
                    X_test, Y_test, plotting, files = data_generator_test(test_folder, seq_len, seq_len-1, phase =phase, val_mask=variable_set[0])
                    
                    if "_norm" in val_mask_name:
                        X_train[:,:,0] = torch.linalg.norm(X_train[:,:,0:3], dim = 2)
                        X_train[:,:,1] = torch.linalg.norm(X_train[:,:,3:6], dim = 2)
                        X_train = X_train[:,:,[0,1]+list(range(6,X_train.shape[2],1))]

                    if "_derivative" in val_mask_name:
                        X_train = torch.from_numpy(np.gradient(X_train, axis=1))
                    
                    augment =  "augment" in val_mask_name
                    print("augment: ", augment)

                    m = X_train.mean(0, keepdim=True)
                    s = X_train.std(0, unbiased=False, keepdim=True)
                    X_train -= m
                    X_train /= s

                    X_train = X_train.permute(0,2,1)

                    for traj_idx in range(len(X_test)):
                        if "_norm" in val_mask_name:
                            X_test[traj_idx][:,:,0] = torch.linalg.norm(X_test[traj_idx][:,:,0:3], dim = 2)
                            X_test[traj_idx][:,:,1] = torch.linalg.norm(X_test[traj_idx][:,:,3:6], dim = 2)
                            X_test[traj_idx] = X_test[traj_idx][:,:,[0,1]+list(range(6,X_test[traj_idx].shape[2],1))]
                        if "_derivative" in val_mask_name:
                            X_test[traj_idx] = torch.from_numpy(np.gradient(X_test[traj_idx], axis=1))
                        X_test[traj_idx] -= m
                        X_test[traj_idx] /= s
                        X_test[traj_idx] = X_test[traj_idx].permute(0,2,1)

                    print(X_train.shape)
                    input_channels = X_train.shape[1] # after permute

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
                    model_name = "TCN_bs"+str(batch_size)+"_seed"+str(seed)+"_do"+str(int(dropout*100))+"_seq"+str(seq_len)+"_"+val_mask_name+"_phase"+str(phase)
                    model_folder = "models_3/"+model_name
                    Path(model_folder).mkdir(parents=True, exist_ok=True)
                    FORMAT = '%Y-%m-%-d_%H%M%S'
                    stamp = datetime.now().strftime(FORMAT)
                    for ep in range(1, epochs+1):
                        train_loss_, train_report = train(ep, X_train, Y_train, augment = augment)
                        train_loss.append(train_loss_)
                        test_loss_, test_report = evaluate_test(X_test, Y_test, plotting, files, "log/evaluate_inputs/"+model_name+"_ep"+str(ep)+"_"+stamp)
                        test_loss.append(test_loss_)
                        test_reports.append(test_report)
                        train_reports.append(train_report)
                        # Save
                        torch.save(model.state_dict(), model_folder+"/"+model_name+"_ep"+str(ep)+".pt")
                    l = []
                    for key in list(test_report.keys())[:-3]:
                        for key2 in list(test_report.get(key).keys())[:-1]:
                            l.append(key+"_"+key2)
                    l.append("accuracy")        
                    with open('log/evaluate_inputs/train_test_'+model_name+stamp+'.csv', 'w') as f:
                        f.write(f"epoch, train_loss, test_loss, "+', '.join(l)+", train_accuracy"+"\n")
                        for loss in enumerate(zip(train_loss, test_loss, test_reports, train_reports)):
                            l = []
                            for key in list(loss[1][2].keys())[:-3]:
                                for key2 in list(loss[1][2].get(key).keys())[:-1]:
                                    l.append(f"{loss[1][2].get(key).get(key2)}") 
                            l.append(f"{loss[1][2].get('accuracy')}")
                            f.write(f"{loss[0]+1},{loss[1][0]}, {loss[1][1]}, "+', '.join(l)+f",{loss[1][3].get('accuracy')}"+"\n")
