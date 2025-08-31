

#packages are in parent folder
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname(__file__) ) )
import os
#check avalable packages in the current environment
print(sys.path)

#chdir to parent folder
os.chdir( os.path.dirname( os.path.dirname(__file__) ) )
print(os.getcwd())



import time


##


import subprocess
import multiprocessing
from threading import main_thread
from pipeline_utils import   config_study_name,Run
import os
import copy
import random
#time.sleep(60*60*4)

#EXP_FLAG="student"
EXP_FLAG="compute_and_visualize_DE_and_MAD"

resources_dict={"0":1,"1":1,}   #id:load

dataset_to_evaluate=[
                    ("cora",1,1),
                    ("citeseer",1,1),
                    ("pubmed",1,1),
                    ("a-computer",1,1),
                    ("a-photo",1,1),
                    ("ogbn-arxiv",1,1),
                    ("ogbn-products",1,1) 
                ]   # dataset,worker_num,_

prefix="record_DEs";specified_args=["dataset","teacher","exp_setting"]



args_name_mapping={"weight_decay":"wd","teacher":"tch","student":"stu","dw":"dw","feat_distill":"fd","adv":"adv","exp_setting":"setting" }

sample_number=100



fixed_info={}
task_space={
            "exp_setting":["tran","ind"],
            "teacher":["SAGE"],
            "output_path":[
                "./results/outputs_with_learned_MLP"]
            }
# this is for all dataset


def get_tasks(task_space):
    tasks=[{}]
    for k,v in task_space.items():
        tasks=expand_task(tasks,k,v)
    return tasks

def expand_task(tasks,k,v):
    temp_tasks=[]
    if type(k) is tuple:
        for t in tasks:
            temp_t=copy.deepcopy(t)
            
            raise NotImplementedError

            temp_tasks.append(temp_t)
        return temp_tasks
    if type(v) is str and type(eval(v)) is list:
        for value in eval(v):
            if k.startswith("search_"):
                value=str([value])
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    elif type(v) is list:
        for value in v:
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    else:
        for t in tasks:
            temp_t=copy.deepcopy(t)
            temp_t[k]=v
            temp_tasks.append(temp_t)
    return temp_tasks

        
task_to_evaluate_to_sample=get_tasks(task_space)
print(task_to_evaluate_to_sample)
#print how many tasks to evaluate and some samples
print(f"total tasks to evaluate: {len(task_to_evaluate_to_sample)}")
print(f"sample tasks to evaluate: {task_to_evaluate_to_sample[:5]}")
start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#tc=0
resources=resources_dict.keys()
pool=multiprocessing.Queue( sum([  v  for k,v in resources_dict.items()   ])  )
for i in resources:
    for j in range(resources_dict[i]):
        pool.put(i+str(j))
tasks_list=[]
for dataset,cost,_ in dataset_to_evaluate:
    if sample_number:
        task_to_evaluate=random.sample( task_to_evaluate_to_sample,sample_number) if len(task_to_evaluate_to_sample)>sample_number else task_to_evaluate_to_sample
    else:
        task_to_evaluate=task_to_evaluate_to_sample
    for task in task_to_evaluate:
        args_dict={}
        for dict_to_add in [task,fixed_info]:
            for k,v in dict_to_add.items():
                args_dict[k]=v
        args_dict['dataset']=dataset
        study_name,study_storage=config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict,args_name_mapping=args_name_mapping)
        args_dict['study_name']=study_name
        args_dict['cost']=cost
        tasks_list.append(args_dict)

sub_queues=[]
items=len(tasks_list)%60
for i in range(items):
    sub_queues.append(tasks_list[60*i:(60*i+60)])
sub_queues.append(tasks_list[(60*items+60):])

if items==0:
    sub_queues.append(tasks_list)

## split the tasks, or it may exceeds of maximal size of sub-processes of OS.
idx=0
tc=len(tasks_list)
for sub_tasks_list in sub_queues:
    process_queue=[]
    for i in range(len(sub_tasks_list)):
        idx+=1
        p=Run(sub_tasks_list[i],idx=idx,tc=tc,pool=pool,start_time=start_time,EXP_FLAG=EXP_FLAG)
        p.daemon=True
        p.start()
        process_queue.append(p)

    for p in process_queue:
        p.join()
    

print('end all')




end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"Start time: {start_time}\nEnd time: {end_time}\nwith {len(task_to_evaluate)*len(dataset_to_evaluate)} tasks")

