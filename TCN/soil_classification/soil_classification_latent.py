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
from TCN.soil_classification.model_AE import ConvAE, DenseAE
from TCN.soil_classification.model_VAE import ConvVAE, DenseVAE
from TCN.soil_classification.model_classifier import LatentClassifier
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

def train(epoch, X_train, Y_train):
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


if __name__ == "__main__":
    # Define tests
    sequence_length = [128]#[50, 100, 250, 500]
    #variable_set = [list(range(LIN_ACC_X,ANG_VEL_Z+1,1))+list(range(BOOM,PITCH+1,1))]
    variable_set = [list(range(LIN_ACC_X,ANG_VEL_Z+1,1))]
    #variable_set_name = ["imu_no_orient+torques"]
    variable_set_name = ["imu_no_orient_norm_derivative"]
    phase_set = [2]
    for seq_len in sequence_length:
        for val_mask, val_mask_name in zip(variable_set, variable_set_name):
            for phase in phase_set:
                # maximum number of training data
                overlap = seq_len - 1
                n_classes = 4
                print("Producing data...")
                train_folder = "/home/mads/git/TCN/TCN/soil_classification/data/exp_1604/train"
                test_folder = "/home/mads/git/TCN/TCN/soil_classification/data/exp_1604/test"
                X_train, Y_train = data_generator(train_folder, seq_len, overlap, phase =phase, val_mask=val_mask, rng=rng)
                X_test, Y_test, plotting, files = data_generator_test(test_folder, seq_len, overlap, phase=phase, val_mask=val_mask)
                
                if "_norm" in val_mask_name:
                    X_train[:,:,0] = torch.linalg.norm(X_train[:,:,0:3], dim = 2)
                    X_train[:,:,1] = torch.linalg.norm(X_train[:,:,3:6], dim = 2)
                    X_train = X_train[:,:,[0,1]+list(range(6,X_train.shape[2],1))]

                if "_derivative" in val_mask_name:
                    X_train = torch.from_numpy(np.gradient(X_train, axis=1))

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


                #Model paths
                # DenseAE_small = "AE_model/DenseAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/DenseAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep64.pt"
                # DenseAE_large = "AE_model/DenseAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240/DenseAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240_ep64.pt"
                # DenseAE_largeL1 = "AE_model/DenseAEL1_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240/DenseAEL1_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240_ep64.pt"
                # DenseAE_smallL1 = "AE_model/DenseAEL1_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/DenseAEL1_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep57.pt"
                # ConvAE_8 = "AE_model/ConvAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/ConvAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep64.pt"
                # ConvAE_small = "AE_model/ConvAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf5/ConvAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf5_ep64.pt"
                # ConvAE_large = "AE_model/ConvAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240/ConvAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240_ep64.pt"

                # DenseVAE_small = "AE_model/DenseVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/DenseVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep64.pt"
                # DenseVAE_large = "AE_model/DenseVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240/DenseVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf240_ep64.pt"
                # ConvVAE_large = "AE_model/ConvVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf5/ConvVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf5_ep64.pt"
                # ConvVAE_small = "AE_model/ConvVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf60/ConvVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf60_ep64.pt"
                # ConvVAE_8 = "AE_model/ConvVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/ConvVAE_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep64.pt"
                ConvVAE_8_2 = "AE_model/ConvVAE_KL0045rel_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/ConvVAE_KL0045rel_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep64.pt"
                ConvVAED_8_grad = "AE_model/ConvVAED_Test14_bs256_do0_seq128_imu_no_orient_derivative_phase2_nlf8/ConvVAED_Test14_bs256_do0_seq128_imu_no_orient_derivative_phase2_nlf8_ep128.pt"
                ConvVAED_8_norm = "AE_model/ConvVAED_Test13_bs256_do0_seq128_imu_no_orient_norm_phase2_nlf8/ConvVAED_Test13_bs256_do0_seq128_imu_no_orient_norm_phase2_nlf8_ep128.pt"
                ConvVAED_8_norm_grad = "AE_model/ConvVAED_Test16_bs256_do25_seq128_imu_no_orient_norm_derivative_phase2_nlf8/ConvVAED_Test16_bs256_do25_seq128_imu_no_orient_norm_derivative_phase2_nlf8_ep128.pt"


                #ConvVAED_8 = "AE_model/ConvVAED_Test6_bs256_do0_seq128_imu_no_orient_phase2_nlf8/ConvVAED_Test6_bs256_do0_seq128_imu_no_orient_phase2_nlf8_ep64.pt"
                #ConvVAED_8 = "AE_model/ConvVAED_Test11_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8/ConvVAED_Test11_bs256_do0_seq128_imu_no_orient+torques_phase2_nlf8_ep64.pt"
         
                input_channels = X_train.shape[1] # after permute
                channel_sizes = [30,30,30,30,30,4]#[nhid]*6#levels
                kernel_size = ksize
                dropout = args.dropout

                latent_dim = 8
                hidden_dims = [1000,800,500, 240]
                classifier_hidden_dims = [32,32]

                #back_model = DenseAE(input_channels, seq_len, latent_dim, hidden_dims)
                #back_model = ConvAE(input_channels, channel_sizes, kernel_size)
                #back_model = DenseVAE(input_channels, seq_len, latent_dim, hidden_dims)
                back_model = ConvVAE(input_channels, channel_sizes, kernel_size)
                back_model.load_state_dict(torch.load(ConvVAED_8_norm_grad))

                model = LatentClassifier(back_model, latent_dim, classifier_hidden_dims, n_classes, VAE=True, dropout=0.35)
                model.freeze_back()

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
                model_name = "latent_class_3232_bs"+str(batch_size)+"_do"+str(int(dropout*100))+"_seq"+str(seq_len)+"_"+val_mask_name+"_phase"+str(phase)+"_nlf"+str(latent_dim)
                model_folder = "models_latent_class/"+model_name
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                FORMAT = '%Y-%m-%-d_%H%M%S'
                stamp = datetime.now().strftime(FORMAT)
                for ep in range(1, epochs+1):
                    train_loss_, train_report = train(ep, X_train, Y_train)
                    train_loss.append(train_loss_)
                    test_loss_, test_report = evaluate_test(X_test, Y_test, plotting, files, "log/latent_class/"+model_name+"_ep"+str(ep)+"_"+stamp)
                    test_loss.append(test_loss_)
                    test_reports.append(test_report)
                    train_reports.append(train_report)
                    # Save
                    torch.save(model.state_dict(), model_folder+"/"+model_name+"_ep"+str(ep)+".pt")
                l = []
                for key in list(test_report.keys())[:-3]:
                    for key2 in list(test_report.get(key).keys())[:-1]:
                        l.append(key+"_"+key2)
                l.append("val_accuracy")        
                with open('log/latent_class/train_test_'+model_name+stamp+'.csv', 'w') as f:
                    f.write(f"epoch, train_loss, val_loss, "+', '.join(l)+", train_accuracy"+"\n")
                    for loss in enumerate(zip(train_loss, test_loss, test_reports, train_reports)):
                        l = []
                        for key in list(loss[1][2].keys())[:-3]:
                            for key2 in list(loss[1][2].get(key).keys())[:-1]:
                                l.append(f"{loss[1][2].get(key).get(key2)}") 
                        l.append(f"{loss[1][2].get('accuracy')}")
                        f.write(f"{loss[0]+1},{loss[1][0]}, {loss[1][1]}, "+', '.join(l)+f",{loss[1][3].get('accuracy')}"+"\n")
