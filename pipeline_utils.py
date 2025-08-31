
import random
import queue
import time
import subprocess
import multiprocessing
from threading import main_thread
import os
import pandas as pd
import csv
import copy
import json 

   

 




def get_best_from_overall_json(overall_json,teacher,setting,model_name,dataset,args_concerned):
    
    if overall_json:
        #load json
        with open(overall_json,"r") as f:
            overall_dict=json.load(f)
        #get the best hyper
        if setting=="ind":
            exp_setting="inductive"
        elif setting=="tran":
            exp_setting="transductive"
        best_hypers=overall_dict[f"{teacher}-{exp_setting}"][model_name][dataset]
        best_hypers={k:v for k,v in best_hypers.items() if k in args_concerned}
        return [best_hypers]
    else:
        return []











def get_tasks_linear_around(task_space,best_hyper):
    tasks=[]
    for param_in_space,param_values in task_space.items():
        assert param_in_space in best_hyper.keys()
        for param_value in param_values:
            temp_t={}
            #copy best hyper except specified param
            for param_in_best,value_in_best in best_hyper.items():
                if "search_" in param_in_best:
                    
                    temp_t[param_in_best]= f"[{value_in_best}]"  if param_in_best!=param_in_space else f"[{param_value}]" 
                else:
                    
                    temp_t[param_in_best]=value_in_best if param_in_best!=param_in_space else param_value
            tasks.append(temp_t)

    
    return tasks


def get_tasks(task_space):
    tasks=[{}]
    for k,v in task_space.items():
        tasks=expand_task(tasks,k,v)
    return tasks

def expand_task(tasks,k,v):
    temp_tasks=[]
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



def proc_yes(yes,args_dict):
    temp_yes=[]
    for name in yes:
        temp_yes.append(f"{name}_{args_dict[name]}")
    return temp_yes
  
class Run( multiprocessing.Process):
    def __init__(self,task,pool=0,idx=0,tc=0,start_time=0,EXP_FLAG="student"):
        super().__init__()
        self.task=task
        self.log=os.path.join(task['study_name'])
        self.idx=idx
        self.pool=pool
        self.device=None
        self.tc=tc
        self.start_time=start_time
        self.EXP_FLAG=EXP_FLAG
        #self.pbar=pbar
    def run(self):
        #print(f"{'*'*10} study  {self.log} no.{self.idx} waiting for device")
        count=0
        device_units=[]
        while True:
            if len(device_units)>0:
                try:
                    unit=self.pool.get(timeout=10*random.random())
                except queue.Empty:
                    for unit in device_units:
                        self.pool.put(unit)
                    print(f"Hold {str(device_units)} and waiting for too long! Throw back and go to sleep")
                    time.sleep(100*random.random())
                    device_units=[]
                    count=0
                    continue
            else:
                unit=self.pool.get()
            if len(device_units)>0:  # consistency check
                if unit[0]!=device_units[-1][0]:
                    print(f"Get {str(device_units)} and {unit} not consistent devices and throw back it")
                    self.pool.put(unit)
                    time.sleep(100*random.random())
                    continue
            count+=1
            device_units.append(unit)
            if count==self.task['cost']:
                break


        print(f"{'-'*10}  study  {self.log} no.{self.idx} get the devices {str(device_units)} and start working")
        self.device=device_units[0][0]
        try:
            exit_command=get_command_from_argsDict(self.task,self.device,self.idx,self.EXP_FLAG)
            
            print(f"running: no.{self.idx}")
            subprocess.run(exit_command,shell=True)
            print(exit_command)
        finally:
            for unit in device_units:
                self.pool.put(unit)
            #localtime = time.asctime( time.localtime(time.time()) )
        
        end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Start time: {self.start_time}\nEnd time: {end_time}\nwith {self.idx}/{self.tc} tasks")

        print(f"  {'<'*10} end  study  {self.log} no.{self.idx} of command ")



def get_command_from_argsDict(args_dict,gpu,idx,EXP_FLAG="student"):
    if EXP_FLAG=="student":
        command='unbuffer python -W ignore train_student.py  '
    elif EXP_FLAG=="teacher":
        command='unbuffer python -W ignore train_teacher.py  '
    elif EXP_FLAG=="compute_and_visualize_DE_and_MAD":
        command='python analyze_teacher_info.py  '
    elif EXP_FLAG=="compute_and_visualize_DE_and_MAD_student":
        command='python analyze_student_info.py  ' 
    for key in args_dict.keys():
        command+=f" --{key} {args_dict[key]} "


    command+=f" --device {gpu} "
    if os.name!="nt": # linux
        #time format=yyyymmdd
        date=time.strftime("%Y%m%d", time.localtime())
        time.sleep(1*random.random())
        if not os.path.exists(f"./log"):
            os.mkdir(f"./log")
        if not os.path.exists(f"./log/{date}"):
            os.mkdir(f"./log/{date}")
        command+=f"   >> ./log/{date}/{args_dict['study_name']}.txt 2>&1 "
    return command






def config_study_name(prefix,specified_args,extract_dict,args_name_mapping):
    study_name=prefix
    for k in specified_args:
        if k not in extract_dict.keys():
            print(f"whole args: {extract_dict}")
            raise Exception(f"specified arg {k} not in extract_dict")
        v=extract_dict[k]
        if k in args_name_mapping:
            k=args_name_mapping[k]
        study_name+=f"_{k}_{v}"
    if study_name[0]=="_":
        study_name=study_name.replace("_","",1)
    study_storage=f"sqlite:///db/{study_name}.db"
    return study_name,study_storage



 
def get_tasks(task_space):
    tasks=[{}]
    for k,v in task_space.items():
        tasks=expand_task(tasks,k,v)
    return tasks

def expand_task(tasks,k,v):
    # v is a list
    temp_tasks=[]
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

def expand_task_with_params(task_list,params):
    if len(params)==0:
        return task_list
    else:
        temp_tasks=[]
        for t in task_list:
            for p in params:
                temp_t=copy.deepcopy(t)
                temp_t.update(p)
                temp_tasks.append(temp_t)
        return temp_tasks




def conduct_exp(resources_dict,tasks_list,start_time,EXP_FLAG):
        
    resources=resources_dict.keys()
    pool=multiprocessing.Queue( sum([  v  for k,v in resources_dict.items()   ])  )
    for i in resources:
        for j in range(resources_dict[i]):
            pool.put(i+str(j))
    print(f"total tasks to evaluate: {len(tasks_list)}")

    sub_queues=[]
    pool_len=pool.qsize()
    items=len(tasks_list)% (4*pool_len)
    for i in range(items):
        sub_queues.append(tasks_list[(4*pool_len)*i:((4*pool_len)*i+(4*pool_len))])
    sub_queues.append(tasks_list[((4*pool_len)*items+(4*pool_len)):])

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
    print(f"Start time: {start_time}\nEnd time: {end_time}\nwith {len(tasks_list)} tasks")

