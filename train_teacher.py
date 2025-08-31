import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data
import sys
arg=sys.argv[1:]
print(f"""captured args are {arg}""")
import warnings
warnings.filterwarnings('ignore')
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import run_transductive, run_inductive


import dgl
import csv

def get_mean_and_std(scores):
    scores_names=scores[0].keys()
    scores_names=list(scores_names)
    scores_mean={}
    scores_std={}
    for name in scores_names:
        scores_mean[name]=[]
        scores_std[name]=[]
    for score in scores:
        for name in scores_names:
            scores_mean[name].append(score[name])
            scores_std[name].append(score[name])
    for name in scores_names:
        scores_mean[name]=np.array(scores_mean[name]).mean()
        scores_std[name]=np.array(scores_std[name]).std()
    return scores_mean, scores_std

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        #action="store_true",
        type=str2bool,
        default=True,
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=False, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        #action="store_true",
        type=str2bool,
        default=True,
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    
    parser.add_argument(
        "--save_learned_MLP_info",
        type=str2bool,
        default=False,
        help="Set to True to save the learned MLP layers in the model, and the features before and after graph aggregatio will also be saved",
    )

    parser.add_argument(
        "--save_graph_aggregation",
        type=str2bool,
        default=False,
        help="Set to True to save the features before and after graph aggregatio",
    )

    parser.add_argument(
        "--save_teacher_layer_info",
        type=str2bool,
        default=False,
        help="""Set to True to save all the hidden features, it's noted that all occured feature matrix including before and after graph aggregation will be saved, and including before and after the MLP part, the format should be:
        data={
            "transformation_type_by_order": list of str "GA" and "MLP",
                *example: ["GA", "MLP", "GA", "MLP"], or ["GA", "MLP","MLP","GA","GA","MLP"], each consecutive two elements must contain one "GA" and one "MLP".
            "MLPs_by_order": list of MLP weight and bias tensors, 
                *example: [MLP_1_weight, MLP_1_bias, MLP_2_weight, MLP_2_bias], or [ MLP_1_weight, MLP_1_bias, MLP_2_weight, MLP_2_bias, MLP_3_weight, MLP_3_bias] each consecutive two elements must contain one MLP weight and one MLP bias.
            "feature_matrices_by_order": list of feature matrices, 
                *example: [input_feature, hidden_feature_1, hidden_feature_2, hidden_matrix_3, hidden_matrix_4], it is starting from one input feature matrix, after that, each consecutive two elements must contain one feature matrix before graph aggregation and one feature matrix after graph aggregation, which is corresponding to the transformation_type_by_order.
            }
        """,
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
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

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        # default="./tran.conf.yaml",
        #default=".conf.yaml",
        default=None,
        help="Path to model configeration",
    )
    #fixed_arg
    parser.add_argument("--fixed_arg",type=str, default=None, help="specify the param name which use the value in args not in conf file")
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )
    #num_heads
    parser.add_argument("--num_heads" ,type=int, default=8)
    parser.add_argument("--attn_dropout_ratio", type=float, default=0)

    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--residual', type=str, default="True") 
    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--K", type=int, default=10) # used for appnp
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        #action="store_true",
        type=str2bool,
        default=False,
        help="Set to True to compute and store the min-cut loss",
    )
    
    #for study 
    
    parser.add_argument( "--study_name", type=str, default="temp"    )
    parser.add_argument( "--cost", type=int, default=1    )
    
    

    args = parser.parse_args()
 
    return args


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    # print('args.seed: ', args.seed)
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_noise != 0:
        if "noisy_features" not in str(args.output_path):
            args.output_path = Path.cwd().joinpath(
                args.output_path, "noisy_features", f"noise_{args.feature_noise}"
            )

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    
    
    
    """ Load data """
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        args,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )
    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    args.label_dim = labels.int().max().item() + 1 
        

    if 0 < args.feature_noise <= 1:
        feats = (
                        1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)
    

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.exp_setting + args.model_config_path, args.teacher, args.dataset)
    else:
        conf["model_name"]=args.teacher
    conf = dict(args.__dict__, **conf)

    if args.fixed_arg is not None:

        conf[args.fixed_arg] = args.__dict__[args.fixed_arg]
    
    conf["device"] = device
    # filter list in conf
    #logger.info(f"conf: {    [   (k, v) for k, v in conf.items() if v is list]    }")
    conf_for_print={}
    for k, v in conf.items():
        if type(v) is list:
            continue
        else:
            conf_for_print[k]=v
    logger.info(f"conf: {conf_for_print}")

    """ Model init """
    model = Model(conf, args,graph=g)
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = torch.nn.NLLLoss()  
    evaluator = get_evaluator(conf["dataset"])

    """ Data split and run """
    loss_and_score = []
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)

        out, score_val, score_test, emb_list = run_transductive(
                args,
                conf,
                model,
                g,
                feats,
                labels,
                indices,
                criterion,
                evaluator,
                optimizer,
                logger,
                loss_and_score,
            )
        score_lst = [score_test]

    elif args.exp_setting == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)

        out, score_val, score_test_tran, score_test_ind, emb_list = run_inductive(
                args,
                conf,
                model,
                g,
                feats,
                labels,
                indices,
                criterion,
                evaluator,
                optimizer,
                logger,
                loss_and_score,
            )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving teacher outputs """
    if not args.compute_min_cut:
        if 'MLP' not in model.model_name:
            out_np = out.detach().cpu().numpy()
            np.savez(output_dir.joinpath("out"), out_np)
            out_emb_list = emb_list[-1].detach().cpu().numpy()  # last hidden layer
            np.savez(output_dir.joinpath("out_emb_list"), out_emb_list)

        """ Saving loss curve and model """
        if args.save_results:
            # Loss curves
            loss_and_score = np.array(loss_and_score)
            np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

            # Model
            torch.save(model.state_dict(), output_dir.joinpath("model.pth"))
  
        if args.save_learned_MLP_info:
            # save the learned MLP layers in the model, and the features before and after graph aggregatio will also be saved
            print('save_learned_MLP_info------')
            # save the learned MLP layers in the model
            MLP_dir=output_dir.joinpath("learned_MLP_part.pth")
            #from post_fix to str
            MLP_dir=str(MLP_dir)
            MLP_layers=model.encoder.get_MLP_layers()
            torch.save(MLP_layers, MLP_dir)
        if args.save_graph_aggregation:
            # save the features before and after graph aggregatio
            print('save features before and after graph aggregatio------')
            features_before_aggr_dir=output_dir.joinpath("features_before_aggr.pth")
            features_after_aggr_dir=output_dir.joinpath("features_after_aggr.pth")
            features_before_aggr_list=[x.cpu() for x in model.encoder.get_features_before_aggr()]
            features_after_aggr_list=[x.cpu() for x in model.encoder.get_features_after_aggr()]
            torch.save(features_before_aggr_list, features_before_aggr_dir)
            torch.save(features_after_aggr_list, features_after_aggr_dir)
        if args.save_teacher_layer_info:
            # save 
            """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
            """
            
            teacher_layer_info=model.encoder.get_teacher_layer_info()
            teacher_layer_info_dir=output_dir.joinpath("teacher_layer_info.pth")
            teacher_layer_info_dir=str(teacher_layer_info_dir)
            torch.save(teacher_layer_info, teacher_layer_info_dir)
            





    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut)

    return score_lst


def repeat_run(args):
    if args.exp_setting == "tran":
        scores = []
    elif args.exp_setting == "ind":
        scores_tran = []
        scores_ind = []
    for seed in range(args.num_exp):
        args.seed = seed
        temp_score = run(args)
        #temp_score=temp_score[0]
        if args.exp_setting == "tran":
            scores.append(temp_score[0])
        elif args.exp_setting == "ind":
            scores_tran.append(temp_score[0])
            scores_ind.append(temp_score[1])
    #scores_np = np.array(scores)
    #return scores_np.mean(axis=0), scores_np.std(axis=0)
    #compute mean and std of all kinds of scores k:v pairs
    if args.exp_setting == "tran":
        scores_mean, scores_std = get_mean_and_std(scores)
    elif args.exp_setting == "ind":
        scores_mean_tran, scores_std_tran = get_mean_and_std(scores_tran)
        scores_mean_ind, scores_std_ind = get_mean_and_std(scores_ind)
        scores_mean = {}
        scores_std = {}
        for k,v in scores_mean_tran.items():
            scores_mean[k+'_tran'] = v
        for k,v in scores_std_tran.items():
            scores_std[k+'_tran'] = v
        for k,v in scores_mean_ind.items():
            scores_mean[k+'_ind'] = v
        for k,v in scores_std_ind.items():
            scores_std[k+'_ind'] = v
    return scores_mean, scores_std



def main():
    args = get_args()

    args.time_cost_train_per_epoch=[]
    args.time_cost_eval_per_epoch=[]
    args.memory_cost_peak_train=[]
    args.memory_cost_peak_eval=[]
    
    score_str=""
    if args.num_exp == 1:
        score = run(args)
        #score_str = "".join([f"{s : .4f}\t" for s in score])
        for k, v in score.items():
            score_str += f"{k}: {v : .4f}\t"

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        for k, v in score_mean.items():
            score_str += f"{k}: {v : .4f} +- {score_std[k] : .4f}\t"
 

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")
 
    print(score_str)

    
    args.time_cost_train_per_epoch= np.mean(np.array(args.time_cost_train_per_epoch))
    args.time_cost_eval_per_epoch= np.mean(np.array(args.time_cost_eval_per_epoch))
    if len(args.memory_cost_peak_train)==0:
        args.memory_cost_peak_train=0
    else:
        args.memory_cost_peak_train= max(args.memory_cost_peak_train)
    if len(args.memory_cost_peak_eval)==0:
        args.memory_cost_peak_eval=0
    else:
        args.memory_cost_peak_eval= max(args.memory_cost_peak_eval)
    
    
    concerned_args = ["study_name", "dataset", "exp_setting","time_cost_train_per_epoch","time_cost_eval_per_epoch","memory_cost_peak_train","memory_cost_peak_eval" ]
    
    toCsv={}
    # write all the arg information and results information to csv file
    for arg in vars(args):
        toCsv[arg]=getattr(args, arg)
    for name in score_mean.keys():
        toCsv[name]=score_mean[name]
        toCsv[name+'_std']=score_std[name]
        concerned_args.append(name)
        
    new_header=list(toCsv.keys())
    rows_dicts=[]
    # record the previous results
    if  args.output_dir.parent.joinpath("exp_results.csv").exists():
        with open(args.output_dir.parent.joinpath("exp_results.csv"), 'r') as f:
            reader = csv.reader(f)
            
            rows = [row for row in reader]
            old_header=rows[0]
            for row in rows[1:]:
                d={}
                for i in range(len(old_header)):
                    d[old_header[i]]=row[i]
                rows_dicts.append(d)
        # write the new results and the previous results to csv file, if the arg name in new results is not in previous results,  add it to previous results with value ''
        for arg_name in new_header:
            if arg_name not in old_header:
                old_header.append(arg_name)
                # rows_dict
                for row_dict in rows_dicts:
                    row_dict[arg_name]=''
    # write the new results to rows_dicts
    rows_dicts.append(toCsv)
    # write the rows_dicts to csv file with a sorted header
    new_header.sort()
    # let the concerned args be the front columns
    for arg_name in concerned_args:
        new_header.remove(arg_name)
        new_header.insert(0, arg_name)
    with open(args.output_dir.parent.joinpath("exp_results.csv"), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=new_header)
        writer.writeheader()
        for row_dict in rows_dicts:
            writer.writerow(row_dict)


if __name__ == "__main__":
    main()
