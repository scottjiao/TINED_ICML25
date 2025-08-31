import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data, load_out_t, load_out_emb_t, load_learned_MLP_layers, load_learned_graph_aggregation
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import distill_run_transductive, distill_run_inductive
import networkx as nx
from position_encoding import DeepWalk
import dgl
#file lock
import fcntl

import sys


#autograd.detect_anomaly
torch.autograd.set_detect_anomaly(False)

# get the whole input args, the ' should be " in the input args

arg=sys.argv[1:]
print(f"""captured args are {arg}""")


import csv
torch.set_num_threads(4)
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
    #task_uid
    parser.add_argument("--task_uid", type=str, default=0)
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", type=str2bool, default=False )
    #fixed_arg
    parser.add_argument("--fixed_arg", type=str, default=None )
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        #str2bool
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
        "--save_model_and_curve",
        #str2bool
        type=str2bool,
        default=False,
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
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
        #default=".conf.yaml",
        default=None,  # if none, use all the parameter from args, if specified, use the parameter in the specified yaml file
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model") #MLP,MLP_from_sequence_of_layers,MLP_from_sequence_of_layers_pmp
    parser.add_argument(
        "--num_layers", type=int, default=None, help="Student model number of layers"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Student model hidden layer dimensions",
    )
    parser.add_argument("--dropout_ratio", type=float, default=None)
    parser.add_argument(
        "--norm_type", type=str, default=None, help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=None)
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
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
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
        #str2bool
        type=str2bool,
        default=False,
        help="Set to True to compute and store the min-cut loss",
    )
    #undirected
    #parser.add_argument(
    #    "--undirected",
    #    type=str2bool,
    #    default=False,
    #    help="Set to True to use undirected graph",)
    #preprocess_features
    #parser.add_argument( "--preprocess_features", type=str, default=None) #row_normalize, col_normalize, None

    """Distiall"""
    parser.add_argument(
        "--lamb",
        type=float,
        default=1,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    )
    parser.add_argument(
        "--out_t_path", type=str, default="outputs", help="Path to load teacher outputs"
    )
 



    parser.add_argument(
        "--dw",
        type=str2bool,
        default=False,
        help="Set to True to include deepwalk positional encoding",
    )
    parser.add_argument(
        "--feat_distill",
        type=str2bool,
        default=False,
        help="Set to True to include feature distillation loss",
    )
    parser.add_argument(
        "--adv",
        type=str2bool,
        default=False,
        help="Set to True to include adversarial feature learning",
    )

    """dw_walk_length: 50
    dw_num_walks: 3
    dw_window_size: 5
    dw_iter: 5
    dw_emb_size: 16
    adv_eps: 0.045
    feat_distill_weight: 0.1"""
    parser.add_argument( "--dw_walk_length", type=int, default=None    )
    parser.add_argument( "--dw_num_walks", type=int, default=None     )
    parser.add_argument( "--dw_window_size", type=int, default=None     )
    parser.add_argument( "--dw_iter", type=int, default=None     )
    parser.add_argument( "--dw_emb_size", type=int, default=None     )
    parser.add_argument( "--adv_eps", type=float, default=None     )
    parser.add_argument( "--feat_distill_weight", type=float, default=None     )

    """parameter sensitivity"""
    parser.add_argument(
        "--sensitivity_adv_eps",
        type=float,
        default=-1,
        help="adv_eps for parameter sensitivity",
    )
    parser.add_argument(
        "--sensitivity_dw_emb_size",
        type=int,
        default=-1,
        help="dw_emb_size for parameter sensitivity",
    )
    parser.add_argument(
        "--sensitivity_feat_distill_weight",
        type=float,
        default=-1,
        help="feat_distill_weight for parameter sensitivity",
    )
    



    # from_learned_MLP
    parser.add_argument("--from_learned_MLP", type=str2bool, default=False)
    # from_learned_MLP
    parser.add_argument("--learned_MLP_lr_ratio", type=float, default=0)   
    #from_MLP_mode
    parser.add_argument("--from_MLP_mode", type=str, default="gap") #gap, simple, stack_after, stack_before, learned_graph_aggregation,same_as_teacher
    #from_MLP_mode
    parser.add_argument("--GA_init_type", type=str, default="random") #random, indentity
    

    #save_student_layer_info
    parser.add_argument("--save_student_layer_info", type=str2bool, default=False)

    # load_disabled for ablation
    parser.add_argument("--load_disabled", type=str2bool, default=False)

    


    

    #DE_regularization
    parser.add_argument("--DE_regularization", type=str2bool, default=False)
    #DE_regularization_rate
    parser.add_argument("--DE_regularization_rate", type=float, default=0.01)
    #DE_mode
    parser.add_argument("--DE_mode", type=str, default="same_as_teacher") # same_as_teacher, target, ones
    #DE_target
    parser.add_argument("--DE_target", type=str, default=None)
    #DE_log
    parser.add_argument("--DE_log", type=str2bool, default=False)
    #DE_sampling_ratio
    parser.add_argument("--DE_sampling_ratio", type=float, default=1)


    #for study 
    
    parser.add_argument( "--study_name", type=str, default="temp"    )
    parser.add_argument( "--cost", type=int, default=1    )
 


    # sqrt_DER
    parser.add_argument("--sqrt_DER", type=str2bool, default=True)



    args = parser.parse_args()

    print(f"adv_eps: {args.adv_eps}")
    return args


global_trans_dw_feature = None


def get_features_dw(adj, device, is_transductive, args):
    if args.dataset == 'ogbn-products' or args.dataset == 'ogbn-arxiv':
        print('getting dw for ogbn-arxiv/ogbn-products ...')
        G = adj
    else:
        adj = np.asarray(adj.cpu())
        G = nx.Graph(adj)

    model_emb = DeepWalk(G, walk_length=args.dw_walk_length, num_walks=args.dw_num_walks, workers=1)
    model_emb.train(window_size=args.dw_window_size, iter=args.dw_iter, embed_size=args.dw_emb_size)

    emb = model_emb.get_embeddings()  # get embedding vectors
    embeddings = []
    for i in range(len(emb)):
        embeddings.append(emb[i])
    embeddings = np.array(embeddings)
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    if is_transductive:
        global global_trans_dw_feature
        global_trans_dw_feature = embeddings
    else:  # inductive
        pass  # we don't have global_ind_dw_feature since each time seed (data split) is different.
    return embeddings


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
     
    #pmp_info={}

    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
        # only use one gpu
        torch.cuda.set_device(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    else:
        device = "cpu"
    if args.verbose:
        print("Device: ", device)
        torch.autograd.set_detect_anomaly(True)

    if args.feature_noise != 0:
        if "noisy_features" not in str(args.output_path):
            args.output_path = Path.cwd().joinpath(
                args.output_path, "noisy_features", f"noise_{args.feature_noise}"
            )
        # Teacher is assumed to be trained on the same noisy features as well.
        # args.out_t_path = args.output_path

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        dw_emb_path = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            # "dw_emb.pt"
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
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
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
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
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"out_t_dir: {out_t_dir}")

    """ Load data and model config"""
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

    g = g.to(device)
     
    
    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    #args.label_dim = labels.int().max().item() + 1

    args.label_dim = labels.int().max().item() + 1 
    if 0 < args.feature_noise <= 1:
        feats = (
                        1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)
    





    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        # you must specify student name!
        conf = get_training_config(
            # args.model_config_path, args.student, args.dataset
            args.exp_setting + args.model_config_path, args.student, args.dataset
        )  # Note: student config
        conf["model_name"]=args.student
    else:
        conf["model_name"]=args.student
        conf["dataset"]=args.dataset
        
    if args.fixed_arg is not None:

        conf[args.fixed_arg] = args.__dict__[args.fixed_arg]
    # print('conf: ', conf)

    # use parameters from conf
    #if 'dw_walk_length' in conf and 'dw_walk_length' not in args:
        #args.dw_walk_length = conf['dw_walk_length']
    if 'dw_walk_length' in conf and args.dw_walk_length == None :
        args.dw_walk_length = conf['dw_walk_length']
    #if 'dw_num_walks' in conf and 'dw_num_walks' not in args:
        #args.dw_num_walks = conf['dw_num_walks']
    if 'dw_num_walks' in conf and args.dw_num_walks == None:
        args.dw_num_walks = conf['dw_num_walks']
    #if 'dw_window_size' in conf and 'dw_window_size' not in args:
        #args.dw_window_size = conf['dw_window_size']
    if 'dw_window_size' in conf and args.dw_window_size == None:
        args.dw_window_size = conf['dw_window_size']
    #if 'dw_iter' in conf and 'dw_iter' not in args:
        #args.dw_iter = conf['dw_iter']
    if 'dw_iter' in conf and args.dw_iter == None:
        args.dw_iter = conf['dw_iter']
    #if 'dw_emb_size' in conf and 'dw_emb_size' not in args:
        #args.dw_emb_size = conf['dw_emb_size']
    if 'dw_emb_size' in conf and args.dw_emb_size == None:
        args.dw_emb_size = conf['dw_emb_size']
    #if args.adv and 'adv_eps' in conf and 'adv_eps' not in args:
        #args.adv_eps = conf['adv_eps']
    if args.adv and 'adv_eps' in conf and args.adv_eps == None:
        args.adv_eps = conf['adv_eps']
    #if args.feat_distill and 'feat_distill_weight' in conf and 'feat_distill_weight' not in args:
        #args.feat_distill_weight = conf['feat_distill_weight']
    if args.feat_distill and 'feat_distill_weight' in conf and args.feat_distill_weight == None:
        args.feat_distill_weight = conf['feat_distill_weight']


    # parameter sensitivity
    if args.adv and args.sensitivity_adv_eps > 0:
        args.adv_eps = args.sensitivity_adv_eps
    if args.dw and args.sensitivity_dw_emb_size > 0:
        args.dw_emb_size = args.sensitivity_dw_emb_size
    if args.feat_distill and args.sensitivity_feat_distill_weight > 0:
        args.feat_distill_weight = args.sensitivity_feat_distill_weight


    # learning
    if args.learning_rate == None:
        if 'learning_rate' in conf:
            args.learning_rate = conf['learning_rate']
        else:
            args.learning_rate = 0.01
            conf['learning_rate'] = 0.01
    if args.weight_decay == None:
        if 'weight_decay' in conf:
            args.weight_decay = conf['weight_decay']
        else:
            args.weight_decay = 0.0005
            conf['weight_decay'] = 0.0005
    if args.dropout_ratio == None:
        if 'dropout_ratio' in conf:
            args.dropout_ratio = conf['dropout_ratio']
        else:
            args.dropout_ratio = 0.5
            conf['dropout_ratio'] = 0.5
    if args.num_layers == None:
        if 'num_layers' in conf:
            args.num_layers = conf['num_layers']
        else:
            args.num_layers = 2
            conf['num_layers'] = 2
    if args.hidden_dim == None:
        if 'hidden_dim' in conf:
            args.hidden_dim = conf['hidden_dim']
        else:
            args.hidden_dim = 64
            conf['hidden_dim'] = 64
    if args.norm_type == None:
        if 'norm_type' in conf:
            args.norm_type = conf['norm_type']
        else:
            args.norm_type = 'none'
            conf['norm_type'] = 'none'
    if args.batch_size == None:
        if 'batch_size' in conf:
            args.batch_size = conf['batch_size']
        else:
            args.batch_size = 512
            conf['batch_size'] = 512

    
    conf = dict(args.__dict__, **conf)
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

    len_position_feature = 0
    if args.exp_setting == "tran":
        idx_l = idx_train
        idx_t = torch.cat([idx_train, idx_val, idx_test])
        distill_indices = (idx_l, idx_t, idx_val, idx_test)

        # position feature (tran)
        if args.dw:
            logger.info(f"process position feature ...... ")
            if args.dataset == 'ogbn-products' or args.dataset == 'ogbn-arxiv':
                """parser.add_argument( "--dw_walk_length", type=int, default=None    )
                    parser.add_argument( "--dw_num_walks", type=int, default=None     )
                    parser.add_argument( "--dw_window_size", type=int, default=None     )
                    parser.add_argument( "--dw_iter", type=int, default=None     )
                    parser.add_argument( "--dw_emb_size", type=int, default=None     )"""
                dw_emb_path = dw_emb_path.joinpath(f"dw_emb_{args.dw_walk_length}_{args.dw_num_walks}_{args.dw_window_size}_{args.dw_iter}_{args.dw_emb_size}.pt")
                try:
                    loaded_dw_emb = torch.load(dw_emb_path).to(device)
                    print('load dw_emb successfully!', flush=True)
                    position_feature = loaded_dw_emb
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                except:
                    print('cannot load dw_emb, now try to calculate it ...... ', flush=True)
                    network_g = g.cpu()
                    network_g = network_g.to_networkx()
                    print('done with network_g')
                    dw_emb = get_features_dw(network_g, device, is_transductive=True, args=args)
                    torch.save(dw_emb, dw_emb_path)
                    print('save dw_emb successfully')
                    position_feature = global_trans_dw_feature
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                    

            # cpf datasets
            elif args.dataset in ["cora", "citeseer", "pubmed","a-photo","a-computer"]: 
                if args.cal_dw_flag:
                    adj = g.adj().to_dense()
                    get_features_dw(adj, device, is_transductive=True, args=args)

                position_feature = global_trans_dw_feature
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)
                

            logger.info(f"process position feature done! ")

    elif args.exp_setting == "ind":
        # Create inductive split
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, args.split_rate, args.seed
        )
        obs_idx_l = obs_idx_train
        obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
        distill_indices = (
            obs_idx_l,
            obs_idx_t,
            obs_idx_val,
            obs_idx_test,
            idx_obs,
            idx_test_ind,
        )

        # position feature (ind)
        if args.dw:  # We need to run it every time since seed (data split) is different.
            # computation optimized for large datasets.
            if args.dataset == 'ogbn-products':
                dw_emb_path = output_dir.joinpath(f"dw_emb_{args.dw_walk_length}_{args.dw_num_walks}_{args.dw_window_size}_{args.dw_iter}_{args.dw_emb_size}.pt")  # need to include the seed in the path
                if not dw_emb_path.exists():
                    # subgraph
                    print("Doing network_g")
                    trained_grapah = dgl.node_subgraph(g, idx_obs.to(device))
                    network_g = trained_grapah.cpu()
                    network_g = network_g.to_networkx()
                    print('done with network_g')
                    position_feature_obs = get_features_dw(network_g, device, is_transductive=True, args=args)
                    print("done with dw_obs")
                    position_feature_obs = position_feature_obs.cpu()
                    #position_feature = torch.zeros(g.adj().shape[0], position_feature_obs.shape[-1], dtype=torch.float32)

                    # change the order of position_feature_obs
                    idx_position_feature = idx_obs.tolist()
                    #position_feature_list_correct_order = [[] for i in range(g.adj().shape[0])]
                    position_feature_list_correct_order=torch.zeros(g.adj().shape[0], position_feature_obs.shape[-1], dtype=torch.float32)
                    for idx_from_zero, idx_p_f in enumerate(idx_position_feature):
                        temp_position_feature = position_feature_obs[idx_from_zero]
                        #position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)
                        position_feature_list_correct_order[idx_p_f]=temp_position_feature

                    # get the neighbor for every node
                    print("get the neighbor for every node")
                    src_node, dst_node = g.edges()
                    src_node = src_node.cpu().tolist()
                    dst_node = dst_node.cpu().tolist()
                    assert len(src_node) == len(dst_node)
                    idx_test_ind_neighbor_dict = {}
                    idx_test_ind_list = idx_test_ind.tolist()
                    for i in range(len(src_node)):
                        src_node_i = src_node[i]
                        dst_node_i = dst_node[i]
                        if src_node_i not in idx_test_ind_neighbor_dict:
                            idx_test_ind_neighbor_dict[src_node_i] = []
                        idx_test_ind_neighbor_dict[src_node_i].append(dst_node_i)
                        if dst_node_i not in idx_test_ind_neighbor_dict:
                            idx_test_ind_neighbor_dict[dst_node_i] = []
                        idx_test_ind_neighbor_dict[dst_node_i].append(src_node_i)
                        
                    print("doing get the dw for test nodes")
                    # get the dw for test nodes
                    for idx_cur_node_id in idx_test_ind_list:
                        try:
                            idx_cur_node_id_neighbor = idx_test_ind_neighbor_dict[idx_cur_node_id]
                            if len(idx_cur_node_id_neighbor):
                                temp_position_feature = torch.mean(position_feature_list_correct_order[idx_cur_node_id_neighbor, :], dim=0)
                            else:
                                temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                        except:
                            temp_position_feature = np.zeros(position_feature_obs.shape[-1])

                        
                        #position_feature_list_correct_order[idx_cur_node_id]=temp_position_feature
                        #TypeError: can't assign a numpy.ndarray to a torch.FloatTensor
                            
                        position_feature_list_correct_order[idx_cur_node_id]=torch.tensor(temp_position_feature, dtype=torch.float32)

                            
                    position_feature =position_feature_list_correct_order
                    len_position_feature = position_feature.shape[-1]
                    position_feature = position_feature.to(device)
                    feats = torch.cat([feats, position_feature], dim=1)
                    
                    print(f'done with dw for ogbn-products, size of position_feature: {position_feature.shape}')
                    torch.save(position_feature, dw_emb_path)
                    del position_feature_obs, position_feature,position_feature_list_correct_order  # save memory
                else:
                    position_feature = torch.load(dw_emb_path).to(device)
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                    print(f'load dw_emb successfully, size of position_feature: {position_feature.shape}')

            # not computation-friendly for large datasets (e.g., ogbn-products).
            elif args.dataset == 'ogbn-arxiv':
                dw_emb_path = output_dir.joinpath(f"dw_emb_{args.dw_walk_length}_{args.dw_num_walks}_{args.dw_window_size}_{args.dw_iter}_{args.dw_emb_size}.pt")  # include the seed in the path
                if not dw_emb_path.exists():
                    # subgraph
                    trained_grapah = dgl.node_subgraph(g, idx_obs.to(device))
                    network_g = trained_grapah.cpu()
                    network_g = network_g.to_networkx()
                    # print('done with network_g')
                    position_feature_obs = get_features_dw(network_g, device, is_transductive=True, args=args)
                    # print('save dw_emb successfully')
                    position_feature_obs = position_feature_obs.cpu()

                    # change the order of position_feature_obs
                    idx_position_feature = idx_obs.tolist()
                    position_feature_list_correct_order = [[] for i in range(g.adj().shape[0])]
                    for idx_from_zero, idx_p_f in enumerate(idx_position_feature):  # tqdm(
                        temp_position_feature = position_feature_obs[idx_from_zero]
                        position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                    # get the dw for test nodes
                    for idx_cur_node_id in idx_test_ind.tolist():  # tqdm(
                        temp_position_feature = None
                        counter_neighbor_in_obs = 0
                        _, idx_one_in_cur_node = g.out_edges(idx_cur_node_id)
                        idx_one_in_cur_node = idx_one_in_cur_node.tolist()
                        for idx_j in idx_one_in_cur_node:
                            if idx_j not in idx_position_feature:
                                continue
                            if temp_position_feature is None:
                                temp_position_feature = np.asarray(position_feature_list_correct_order[idx_j])
                            else:
                                temp_position_feature += np.asarray(position_feature_list_correct_order[idx_j])
                            counter_neighbor_in_obs += 1
                        # for those we could not find a neighbor
                        if temp_position_feature is None:
                            temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                        else:
                            temp_position_feature /= counter_neighbor_in_obs
                        position_feature_list_correct_order[idx_cur_node_id].extend(temp_position_feature)

                    position_feature = torch.tensor(position_feature_list_correct_order, dtype=torch.float32).to(device)
                    torch.save(position_feature, dw_emb_path)
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                    
                else:
                    position_feature = torch.load(dw_emb_path).to(device)
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                    
                    print(f'load dw_emb successfully, size of position_feature: {position_feature.shape}')

            # cpf dataset
            elif args.dataset in ["cora", "citeseer", "pubmed","a-photo","a-computer"]:
                adj = g.adj().to_dense()
                adj_obs = adj[idx_obs, :][:, idx_obs]

                # take dw from neighbors
                position_feature_obs = get_features_dw(adj_obs, device, is_transductive=False, args=args).cpu()

                idx_position_feature = idx_obs.tolist()
                # change the order of position_feature_obs
                # since idx_obs is not in order, we need to change the order of position_feature_obs
                position_feature_list_correct_order = [[] for i in range(len(adj))]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # fill in the dw for test nodes
                adj_numpy = adj.cpu().numpy()
                for idx_cur_node_id in idx_test_ind.tolist():
                    temp_position_feature = None
                    counter_neighbor_in_obs = 0
                    idx_one_in_cur_node = np.where(adj_numpy[idx_cur_node_id] == 1)[0]  # find the neighbors
                    idx_one_in_cur_node = idx_one_in_cur_node.tolist()  # convert to list
                    # find the neighbors in idx_obs
                    for idx_j in idx_one_in_cur_node:
                        if idx_j not in idx_position_feature:
                            #  this neighbor is not in idx_obs
                            continue
                        if temp_position_feature is None:
                            temp_position_feature = np.asarray(position_feature_list_correct_order[idx_j])
                        else:
                            temp_position_feature += np.asarray(position_feature_list_correct_order[idx_j])
                        counter_neighbor_in_obs += 1
                    # for those we could not find a neighbor, use zero vector
                    if temp_position_feature is None:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    else:
                        # average among neighbors
                        temp_position_feature /= counter_neighbor_in_obs
                    position_feature_list_correct_order[idx_cur_node_id].extend(temp_position_feature)

                position_feature = torch.tensor(position_feature_list_correct_order, dtype=torch.float32).to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)


    
    from_learned_MLP_params={}
    if args.from_learned_MLP:
        learned_MLP_layers=load_learned_MLP_layers(out_t_dir,device)
        learned_MLP_layers=learned_MLP_layers.to(device)
        from_learned_MLP_params['learned_MLP_layers']=learned_MLP_layers   # list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]

        #shape_sequence_of_teacher_layers inferenced from learned_MLP_layers : [ [m1shape1,m1shape2], [m2shape1,m2shape2], ...]
        shape_sequence_of_teacher_layers=[]
        for i in range(len(learned_MLP_layers)):
            if i%2==0:
                shape_sequence_of_teacher_layers.append(learned_MLP_layers[i].shape)
        from_learned_MLP_params['shape_sequence_of_teacher_layers']=shape_sequence_of_teacher_layers
        args.shape_sequence_of_teacher_layers=shape_sequence_of_teacher_layers
        from_learned_MLP_params['from_MLP_mode']=args.from_MLP_mode
        if args.from_MLP_mode in ['learned_graph_aggregation'] :
            from_learned_MLP_params['learned_graph_aggregation']=load_learned_graph_aggregation(out_t_dir,device)
        if args.from_MLP_mode in ["same_as_teacher", "same_as_teacher_appnp"]:
            from_learned_MLP_params['learned_graph_aggregation']=None
    """ Model init """
    model = Model(conf, args, len_position_feature,graph=g,from_learned_MLP_params=from_learned_MLP_params)
    # conf is to set the model parameters, args is to set the training parameters
    model = model.to(device)
    
    

    
    
    if args.from_learned_MLP== True:
        params_dict=[]
        
        param_not_in_learned_MLP=[]
        for name, param in model.named_parameters():
            if id(param) not in list(map(id,model.encoder.learned_MLP_layers.parameters())):
                param_not_in_learned_MLP.append(param)
        params_dict.append({'params': param_not_in_learned_MLP, 'lr': args.learning_rate, 'weight_decay': args.weight_decay})
        params_dict.append({'params': model.encoder.learned_MLP_layers.parameters(), 'lr': args.learning_rate*args.learned_MLP_lr_ratio, 'weight_decay': args.weight_decay})
        #print num of tensors in param_not_in_learned_MLP and learned_MLP_layers
        print('num of tensors in param_not_in_learned_MLP: ',len(param_not_in_learned_MLP),flush=True)
        print('num of tensors in learned_MLP_layers: ',len(list(model.encoder.learned_MLP_layers.parameters())),flush=True)
    else:
        params_dict=[ {'params': model.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},]

    optimizer = optim.Adam(
        params_dict
    )
    
    #criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True) 
    #evaluator = get_evaluator(conf["dataset"])

    
    criterion_l = torch.nn.NLLLoss()  
    evaluator = get_evaluator(conf["dataset"])

    

    """Load teacher model output"""
    out_t = load_out_t(out_t_dir)
    out_emb_t = load_out_emb_t(out_t_dir)
    out_emb_t = out_emb_t.to(device)
    logger.info(
        f"teacher score on train data: {evaluator(out_t[idx_train], labels[idx_train])}"
    )
    logger.info(
        f"teacher score on val data: {evaluator(out_t[idx_val], labels[idx_val])}"
    )
    logger.info(
        f"teacher score on test data: {evaluator(out_t[idx_test], labels[idx_test])}"
    ) 

        

    """Data split and run"""
    loss_and_score = []
    if args.exp_setting == "tran":
        out, score_val, score_test = distill_run_transductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            out_emb_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            g,
            args, 
            from_learned_MLP_params=from_learned_MLP_params,
        )
        score_lst = [score_test]  # score_test: a dictioanry

    elif args.exp_setting == "ind":
        out, score_val, score_test_tran, score_test_ind = distill_run_inductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            out_emb_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            args,
            from_learned_MLP_params=from_learned_MLP_params,
            graph=g,
        )
        score_lst = [score_test_tran, score_test_ind]
    else:
        raise NotImplementedError

    
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")


    #""" Saving loss curve and model """
    if args.save_model_and_curve:
        """ Saving student outputs """
        out_np = out.detach().cpu().numpy()
        np.savez(output_dir.joinpath("out"), out_np)
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    
    if args.save_student_layer_info:
        # save 
        """data=[]
        List of dict, each dict is a transformation, the keys are:
        "transformation_type": str "GA" or "MLP",
        "MLP": MLP weight and bias tensors if transformation_type is "MLP",
        "feature_matrix_in": input feature matrix,
        "feature_matrix_out": output feature matrix,

        The transformation is the same as the order in the model
        """
        
        student_layer_info=model.encoder.get_student_layer_info()
        student_layer_info_dir=output_dir.joinpath("student_layer_info.pth")
        student_layer_info_dir=str(student_layer_info_dir)
            
            

        torch.save(student_layer_info, student_layer_info_dir)
        


    """ Saving min-cut loss"""
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut, flush=True)

    return score_lst


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

def repeat_run(args):
    if args.exp_setting == "tran":
        scores = []
    elif args.exp_setting == "ind":
        scores_tran = []
        scores_ind = []
    for seed in range(args.num_exp):
        if seed == 0:
            cal_dw_flag = True
        else:
            cal_dw_flag = False
        args.cal_dw_flag = cal_dw_flag
        args.seed = seed
        temp_score = run(args)
        #temp_score=temp_score[0]
        #scores.append(temp_score)
        if args.exp_setting == "tran":
            scores.append(temp_score[0])
        elif args.exp_setting == "ind":
            scores_tran.append(temp_score[0])
            scores_ind.append(temp_score[1])
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
    
    if args.num_exp == 1:
        args.cal_dw_flag = True
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])
        if args.exp_setting == 'ind':
            score_prod = score[0] * 0.8 + score[1] * 0.2

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        #score_str = "".join(
        #    [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        #)
        score_str = ""
        for name in score_mean.keys():
            score_str += f"{name}: {score_mean[name] : .4f} +- {score_std[name] : .4f}\t"
            
        #if args.exp_setting == 'ind':
            #score_prod = score_mean[0] * 0.8 + score_mean[1] * 0.2
            #score_prod = score_mean['test_tran'] * 0.8 + score_mean['test_ind'] * 0.2

    #with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
    #    f.write(f"{score_str}\n")

    args.time_cost_train_per_epoch= np.mean(np.array(args.time_cost_train_per_epoch))
    args.time_cost_eval_per_epoch= np.mean(np.array(args.time_cost_eval_per_epoch))
    args.memory_cost_peak_train= max(args.memory_cost_peak_train)
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
    header_changed=False
    rows_dicts=[]
    # record the previous results
    if  args.output_dir.parent.joinpath("exp_results.csv").exists():
        with open(args.output_dir.parent.joinpath("exp_results.csv"), 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            #reader = csv.reader(f)
            
            #csv_reader=csv.reader(csv_file,delimiter=',')
            #to solve _csv.Error: line contains NUL
            reader=csv.reader(x.replace('\0', '') for x in f)
            rows = [row for row in reader]
            old_header=rows[0]
            for row in rows[1:]:
                d={}
                #for i in range(len(old_header)):
                #    d[old_header[i]]=row[i]
                #make more robust, some rows may be abnomal
                for i in range(len(old_header)):
                    if i<len(row):
                        d[old_header[i]]=row[i]
                    else:
                        d[old_header[i]]=''
                rows_dicts.append(d)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        # write the new results and the previous results to csv file, if the arg name in new results is not in previous results,  add it to previous results with value ''
        for arg_name in new_header:
            if arg_name not in old_header:
                header_changed=True
                old_header.append(arg_name)
                # rows_dict
                for row_dict in rows_dicts:
                    row_dict[arg_name]=''
        # if the arg name in previous results is not in new results, let it be '' in new results
        for arg_name in old_header:
            if arg_name not in new_header:
                new_header.append(arg_name)
                toCsv[arg_name]=''
    else:
        # if the csv file does not exist, write the header
        header_changed=True
        
                
    # write the new results to rows_dicts
    rows_dicts.append(toCsv)
    # write the rows_dicts to csv file with a sorted header
    new_header.sort()
    # let the concerned args be the front columns
    for arg_name in concerned_args:
        new_header.remove(arg_name)
        new_header.insert(0, arg_name)
    if header_changed:
        with open(args.output_dir.parent.joinpath("exp_results.csv"), 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=new_header)
            writer.writeheader()
            for row_dict in rows_dicts:
                writer.writerow(row_dict)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    else:
        with open(args.output_dir.parent.joinpath("exp_results.csv"), 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            # only write the new results
            writer = csv.DictWriter(f, fieldnames=new_header)
            row_dict=rows_dicts[-1]
            writer.writerow(row_dict)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)




        

    # for collecting aggregated results
    #print(score_str, flush=True)
    print(f"score_str: {score_str}", flush=True)
    #if args.exp_setting == 'ind':
    #    print('prod: ', score_prod)


if __name__ == "__main__":
    args = get_args()
    main()
