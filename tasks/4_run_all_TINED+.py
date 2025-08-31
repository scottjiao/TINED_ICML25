import time
from typing import Dict, Tuple
import random
import os
import numpy as np
import pandas as pd
#packages are in parent folder
import sys
import os
import json
# to current folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


sys.path.append( os.path.dirname( os.path.dirname(__file__) ) )
sys.path.append( os.path.dirname(__file__) )


#check avalable packages in the current environment
print(sys.path)
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_utils  import  conduct_exp

resources_dict={"0":1,"1":1,}   #id:load
prefix="rep_all_TINED+"

setting_dict={"transductive":"tran","inductive":"ind"}

models_list=["ours+nosmog"]
dataset_list=["cora",
              "citeseer",
              "pubmed","a-computer","a-photo","ogbn-arxiv","ogbn-products"
              ]
settings=["SAGE-transductive","SAGE-inductive",]

specified_info={}


def get_fixed_info(setting,model,dataset):
    fixed_info={
        "dataset":dataset,
        "teacher":setting.split("-")[0],
            "exp_setting":setting_dict[setting.split("-")[1]],
            "out_t_path":"./results/outputs_with_learned_MLP",
            "student":"MLP_from_sequence_of_layers",
            "num_exp":10,
            "patience":50,
            "from_learned_MLP":"True",
            "from_MLP_mode":"same_as_teacher",                 ###
            #"save_student_layer_info":True,
            "DE_regularization":"True",           ###
            "DE_mode":"same_as_teacher",
                    #"dw":"True",
                    #"feat_distill":"True",
                    #"adv":"True",
    }
    if len(model.split("+"))==2:
        fixed_info["dw"]="True"
        fixed_info["feat_distill"]="True"
        fixed_info["adv"]="True"
    else:
        fixed_info["dw"]="False"
        fixed_info["feat_distill"]="False"
        fixed_info["adv"]="False"
    return fixed_info





def get_fn(exp_setting,output_path,dataset,teacher,student):
    
    if  exp_setting  == "tran":
        output_dir = Path.cwd().joinpath(
             output_path ,
            "transductive",
            dataset,
            f"{ teacher }_{ student }",
        )
    elif  exp_setting  == "ind":
        output_dir = Path.cwd().joinpath(
             output_path ,
            "inductive",
            f"split_rate_0.2",
            dataset,
            f"{ teacher }_{ student }",
        )
    result_file_name = output_dir.joinpath(f"exp_results.csv")
    #check if result_file_name exists
    if os.path.exists(result_file_name):
        print(f"result_file_name {result_file_name} exists!")
    return result_file_name


# load f"./best_records/best_records.json"\

with open("./best_records/best_records.json") as f:
    best_records=json.load(f)

# from "./best_records/best_records.json" get the hyperparameters to conduct experiment

tasks_list=[]
for setting in settings:
    
    for model in models_list:
        
        for dataset in dataset_list:
        
        
            exp_setting=setting_dict[setting.split("-")[1]]
            fixed_info=get_fixed_info(setting,model,dataset)
            output_path=f"./results/{prefix}"
            fixed_info["output_path"]=output_path
            study_name=f"{prefix}_{setting}_{model}_{dataset}"
            fixed_info['study_name']=study_name
            fixed_info['cost']=1
            best_hypers=best_records[setting][model][dataset]
            if best_hypers == {}:
                continue
            # if not nosmog
            if model=="ours":
                searching_space_names=[  "DE_sampling_ratio" , "DE_log","squared_DER" ,  "GA_init_type" ,"norm_type" , "learned_MLP_lr_ratio" , "lamb" ,"learning_rate" ,"dropout_ratio" ,"weight_decay","DE_regularization_rate","batch_size","patience","max_epoch"]
            elif model=="ours+nosmog":
                searching_space_names=[  "DE_sampling_ratio" , "DE_log","squared_DER" ,  "GA_init_type" ,"norm_type" , "learned_MLP_lr_ratio" , "lamb" ,"learning_rate" ,"dropout_ratio" ,"weight_decay","DE_regularization_rate","batch_size","patience","max_epoch",
                    "dw_walk_length" ,
                    "dw_num_walks" ,
                    "dw_window_size" ,
                    "dw_iter" ,
                    "dw_emb_size" ,
                    "adv_eps" ,
                    "feat_distill_weight" ]
            else:
                raise ValueError("model not supported")
            # filter the best_hypers from searching_space_names
            best_hypers={k:v for k,v in best_hypers.items() if k in searching_space_names}
            fixed_info.update(best_hypers)

            tasks_list.append(fixed_info)

start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

EXP_FLAG="student"
conduct_exp(resources_dict,tasks_list,start_time,EXP_FLAG)