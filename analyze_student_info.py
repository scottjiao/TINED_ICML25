
import argparse 
from dataloader import load_data
import json

from pathlib import Path
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from utils_gnn_property import compute_dirichlet_energy,compute_mean_average_distance
import dgl

import sys
#print sys args
print(sys.argv[ 1:])
parser = argparse.ArgumentParser(description="Analyze student information")
#output_path
parser.add_argument("--output_path", type=str, default="./results/outputs_deeper_with_learned_MLP_2")
parser.add_argument("--output_fig_path", type=str, default="layer2")
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--teacher", type=str, default="sage")
parser.add_argument("--student", type=str, default="MLP")

parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
)
parser.add_argument("--exp_setting", type=str, default="tran")
parser.add_argument("--split_rate", type=float, default=0.2)

#parser.add_argument("--student", type=str, default="MLP")
parser.add_argument("--num_exps", type=int, default=10)

#lr
parser.add_argument("--lr", type=float, default=0.01)
#wd
parser.add_argument("--weight_decay", type=float, default=5e-4)
#epoch
parser.add_argument("--epochs", type=int, default=1000)
#norm
parser.add_argument("--norm", type=str, default="batch") # batch, layer, none

#study_name
parser.add_argument("--study_name", type=str, default="temp")  
#study_name
parser.add_argument("--cost", type=int, default=1)  
#study_name
parser.add_argument("--device", type=int, default="0")  


args = parser.parse_args()
ratio_DE_array=[]
ratio_MAD_array=[]

loss_by_epoch=None

#print dataset, exp_setting and teacher
#print(args.dataset+" "+args.exp_setting,end="\t")

print(args.output_path,args.output_fig_path)
print( args.dataset+" "+args.exp_setting+" "+args.teacher+" "+args.student,end="\t")

for re in range(args.num_exps):
    seed=re
    #print(f"seed {seed}")
    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{re}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{re}",
        )

    def get_data(output_dir):
        file_name="student_layer_info.pth"
        file_path=os.path.join(output_dir,file_name)
        if os.path.exists(file_path):
            print(f"load {file_path}")
            return torch.load(file_path)
        else:
            # recursively check if parent folder exists
            while True:
                output_dir=output_dir.parent
                if output_dir.exists():
                    print(f"{output_dir} exists")
                    break
                else:
                    print(f"{output_dir} does not exist")
            print(f"file {file_path} does not exist")
            raise FileNotFoundError

    #load student data
    student_layer_info = get_data(output_dir)

    #load whole graph as dgl graph
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        args,
        split_idx=args.split_idx,
        seed=seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )
    """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP", else will get a NoneType object,
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
    student_layer_info

    # flatten and squeeze to standard n*d 2d matrix
    for i in range(len(student_layer_info)):
        for entry in ["feature_matrix_in","feature_matrix_out"]:
            if len(student_layer_info[i][entry].shape )==1:
                # unsqueeze
                student_layer_info[i][entry]=student_layer_info[i][entry].unsqueeze(1)
            if len(student_layer_info[i][entry].shape )==3:
                # squeeze
                student_layer_info[i][entry]=student_layer_info[i][entry].reshape(student_layer_info[i][entry].shape[0],-1)
             
        

    if args.device=="-1":
        device=torch.device("cpu")
    else:
        device=torch.device(f"cuda:{args.device}")
    g=g.to( device)
    for i in range(len(student_layer_info)):
        for entry in ["feature_matrix_in","feature_matrix_out"]:
            student_layer_info[i][entry]=student_layer_info[i][entry].to(device)
            
    DE_ratios=[]
    MAD_ratios=[]
    for i in range(len(student_layer_info)):
        DE_ratios.append(
            compute_dirichlet_energy(g,student_layer_info[i]["feature_matrix_out"]).item()/compute_dirichlet_energy(g,student_layer_info[i]["feature_matrix_in"]).item())
        MAD_ratios.append(
            compute_mean_average_distance(g,student_layer_info[i]["feature_matrix_out"]).item()/compute_mean_average_distance(g,student_layer_info[i]["feature_matrix_in"]).item() )
    #print(f"DEs: {DEs}")
    #print(f"MADs: {MADs}")
    """ratio_DE=[]
    ratio_MAD=[]
    for i in range(len(DEs)):
        if i==0:
            pass
        else:
            ratio_DE.append(DEs[i]/DEs[i-1])
            ratio_MAD.append(MADs[i]/MADs[i-1])"""
    #print(f"ratio_DE: {ratio_DE}")
    ratio_DE_array.append(DE_ratios)
    #print(f"ratio_MAD: {ratio_MAD}")
    ratio_MAD_array.append(MAD_ratios)
    
    #save values to output_dir as json
    DE_fname="DE_ratios.json"
    MAD_fname="MAD_ratios.json"
    print(f"save {DE_fname} and {MAD_fname} to {output_dir}......")
    with open(os.path.join(output_dir,DE_fname),"w") as f:
        json.dump({"DE_ratios":DE_ratios},f)
    with open(os.path.join(output_dir,MAD_fname),"w") as f:
        json.dump({"MAD_ratios":MAD_ratios},f)

#print(student_layer_info["transformation_type_by_order"])
ratio_DE_mean=np.mean(np.array(ratio_DE_array),axis=0)
ratio_DE_std=np.std(np.array(ratio_DE_array),axis=0)
ratio_MAD_mean=np.mean(np.array(ratio_MAD_array),axis=0)
ratio_MAD_std=np.std(np.array(ratio_MAD_array),axis=0)

def print_mean_and_std(name,mean,std):
    s=f"{name}:\t"
    for i in range(len(mean)):
        s+=f"{mean[i]:.2f}+-{std[i]:.2f}"
        if i!=len(mean)-1:
            s+="\t"
    print(s)

#print_mean_and_std("ratio_DE_mean",ratio_DE_mean,ratio_DE_std)
#print_mean_and_std("ratio_MAD_mean",ratio_MAD_mean,ratio_MAD_std)

# get the corresponding idx in list for "MLP" and "GA"
idx_MLP=[]
idx_GA=[]
for i in range(len(student_layer_info)):
    if student_layer_info[i]["transformation_type"]=="MLP":
        idx_MLP.append(i)
    elif student_layer_info[i]["transformation_type"]=="GA":
        idx_GA.append(i)

MLP_ratio_DE_mean=ratio_DE_mean[idx_MLP]
MLP_ratio_DE_std=ratio_DE_std[idx_MLP]
MLP_ratio_MAD_mean=ratio_MAD_mean[idx_MLP]
MLP_ratio_MAD_std=ratio_MAD_std[idx_MLP]

GA_ratio_DE_mean=ratio_DE_mean[idx_GA]
GA_ratio_DE_std=ratio_DE_std[idx_GA]    
GA_ratio_MAD_mean=ratio_MAD_mean[idx_GA]
GA_ratio_MAD_std=ratio_MAD_std[idx_GA]

#print layers
for i in range(len(GA_ratio_DE_mean)):
    print(f"layer {i}",end="\t")
print("")

print_mean_and_std("MLP_ratio_DE",MLP_ratio_DE_mean,MLP_ratio_DE_std)
print_mean_and_std("GA_ratio_DE",GA_ratio_DE_mean,GA_ratio_DE_std)
print_mean_and_std("MLP_ratio_MAD",MLP_ratio_MAD_mean,MLP_ratio_MAD_std)
print_mean_and_std("GA_ratio_MAD",GA_ratio_MAD_mean,GA_ratio_MAD_std)
print("-----------------------------------------------\n")



# errorbar

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
#red dashes for MLP, blue line  for GA
# fig 1 for DE, fig 2 for MAD
# y\in[0,max_y_value+0.1]
# x is the layer number
plt.title(f"{args.dataset} {args.exp_setting} {args.student} DE")
plt.xlabel("layer number")
plt.ylabel("ratio")
# DE
plt.errorbar(idx_MLP,MLP_ratio_DE_mean,yerr=MLP_ratio_DE_std,fmt='r.',ecolor='lightcoral',elinewidth=3,capsize=0)
plt.errorbar(idx_GA,GA_ratio_DE_mean,yerr=GA_ratio_DE_std,fmt='b*',ecolor='lightblue',elinewidth=3,capsize=0)
plt.legend(["MLP","GA"])
# grey background
plt.axvspan(  0, int(len(idx_MLP)+len(idx_GA)-1), facecolor='grey', alpha=0.1)
# compute max_y_value
max_y_value=np.max(np.concatenate((MLP_ratio_DE_mean,GA_ratio_DE_mean)))
try:
    plt.ylim([0,max(max_y_value+0.1,1.1)])
except:
    pass
# horizontal line y=1, color black, linestyle dashed
plt.axhline(y=1, color='k', linestyle='--',alpha=0.5,linewidth=0.3)
#x tick
plt.xticks(np.arange(0, int(len(idx_MLP)+len(idx_GA)), 1.0))

plt.subplot(1,2,2)
plt.title(f"{args.dataset} {args.exp_setting} {args.student} MAD")
plt.xlabel("layer number")
plt.ylabel("ratio")
# MAD
plt.errorbar(idx_MLP,MLP_ratio_MAD_mean,yerr=MLP_ratio_MAD_std,fmt='r.',ecolor='lightcoral',elinewidth=3,capsize=0)
plt.errorbar(idx_GA,GA_ratio_MAD_mean,yerr=GA_ratio_MAD_std,fmt='b*',ecolor='lightblue',elinewidth=3,capsize=0)

plt.legend(["MLP","GA"])
# compute max_y_value
max_y_value=np.max(np.concatenate((MLP_ratio_MAD_mean,GA_ratio_MAD_mean)))

plt.axhline(y=1, color='k', linestyle='--',alpha=0.5,linewidth=0.3)
#x tick
plt.xticks(np.arange(0, int(len(idx_MLP)+len(idx_GA)), 1.0))
try:
    plt.ylim([0,max(max_y_value+0.1,1.1)])
except:
    pass
fig_path=os.path.join("./figs",args.output_fig_path,args.student,args.exp_setting)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
plt.savefig(os.path.join(fig_path,f"{args.dataset}_DE_MAD.png"))
#plt.savefig(f"./figs/{args.dataset}_{args.exp_setting}_{args.student}_DE_MAD.png")




