import numpy as np
import copy
import torch
import dgl
from utils import set_seed
import time
 
from utils import set_seed    


import torch.nn.functional as F





"""
1. Train and eval
"""


def train(model, data, feats, labels, criterion, optimizer, idx_train,args, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    #put model into device
    model.to(labels.device)

    #time
    start_time = time.time()
    #memory
    torch.cuda.reset_peak_memory_stats()
    #torch.cuda.max_memory_allocated()

    # Compute loss and prediction
    if "GCN" in model.model_name or "GAT" in model.model_name or "APPNP" in model.model_name or "SimpleHGN" in model.model_name:
        _, logits = model(data, feats)

    else:
        logits = model(data, feats)
    out = logits.log_softmax(dim=1)  
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #time
    end_time = time.time()
    time_cost=end_time-start_time
    args.time_cost_train_per_epoch.append(time_cost)

    args.memory_cost_peak_train.append(torch.cuda.max_memory_allocated()/1024/1024)
    return loss_val


def train_sage(model, dataloader, feats, labels, criterion, optimizer,args, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = labels.device
    model.train()
    total_loss = 0
    #time
    start_time = time.time()
    #memory
    torch.cuda.reset_peak_memory_stats()

    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)  
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #time
    end_time = time.time()
    args.time_cost_train_per_epoch.append(end_time-start_time)
    args.memory_cost_peak_train.append(torch.cuda.max_memory_allocated()/1024/1024)

    return total_loss / len(dataloader)


def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer,args, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    #time
    start_time = time.time()
    #memory
    torch.cuda.reset_peak_memory_stats()
    #num_batches = max(1, feats.shape[0] // batch_size)
    number_of_samples = feats.shape[0]
    num_batches = int(np.ceil(number_of_samples / batch_size))
    #idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]
    #idx_batch = torch.randperm(number_of_samples)  # shuffle
    # not shuffle
    idx_batch = torch.arange(number_of_samples)
    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch_list=[]
        for i in range(num_batches):
            if (i+1)*batch_size>number_of_samples:
                idx_batch_list.append( idx_batch[i*batch_size:])
            else:
                idx_batch_list.append( idx_batch[i*batch_size:(i+1)*batch_size])
        idx_batch = idx_batch_list
    device = labels.device
    model.to(device)
    total_loss = 0
    for i in range(num_batches):
        batched_feats = feats[idx_batch[i]]
        #_, logits = model(None, feats[idx_batch[i]])
        _,logits = model(None,batched_feats)
        out = logits.log_softmax(dim=1)  

        loss = criterion(out, labels[idx_batch[i]]) 
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if args.verbose:
        print_str=f"Train loss: {total_loss / num_batches:.4f}"
        print(print_str)
    #time
    end_time = time.time()
    args.time_cost_train_per_epoch.append(end_time-start_time)
    args.memory_cost_peak_train.append(torch.cuda.max_memory_allocated()/1024/1024)
    return total_loss / num_batches


def get_PGD_inputs(model, feats, labels, criterion, args):
    iters = 5
    eps = args.adv_eps
    alpha = eps / 4

    # init

    delta = torch.rand(feats.shape) * eps * 2 - eps 
    
    delta = delta.to(feats.device) 
    
    delta = torch.nn.Parameter(delta) 
    

    for i in range(iters):
        p_feats = feats + delta 
        

        _, logits = model(None, p_feats) 
        out = logits.log_softmax(dim=1) 

        loss = criterion(out, labels)
        loss.backward()

        # delta update
        
        delta.data = delta.data + alpha * delta.grad.sign() 
        delta.grad = None  
        delta.data = torch.clamp(delta.data, min=-eps, max=eps) 
        
    output = delta.detach()
        
    return output

 
def train_both_distillation_batch_adv(model, feats, labels, teacher_emb, args, batch_size, criterion, optimizer,
                                      lamb=1,):
    model.train()
    number_of_samples = feats.shape[0]

    #time
    start_time = time.time()
    #memory
    torch.cuda.reset_peak_memory_stats()
    

    num_batches = int(np.ceil(number_of_samples / batch_size))
    #idx_batch = torch.randperm(number_of_samples) # shuffle
    #idx_batch  = torch.randperm(number_of_samples)  # shuffle
    # not shuffle
    idx_batch = torch.arange(number_of_samples)
    #to device
    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch_list=[]
        for i in range(num_batches):
            if (i+1)*batch_size>number_of_samples:
                idx_batch_list.append( idx_batch[i*batch_size:])
            else:
                idx_batch_list.append( idx_batch[i*batch_size:(i+1)*batch_size])
        idx_batch=idx_batch_list
    """if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)"""

    device=labels.device
    total_loss = 0
    for i in range(num_batches):
        batched_feats = feats[idx_batch[i]]

        batch_mlp_emb, logits = model(None, batched_feats) 
        out = logits.log_softmax(dim=1)  

        # adversarial learning
        if args.adv:
            adv_deltas = get_PGD_inputs(model, batched_feats, labels[idx_batch[i]], criterion, args) 
            adv_feats = torch.add(batched_feats, adv_deltas)  
            _, adv_logits = model(None, adv_feats) 
            adv_out = adv_logits.log_softmax(dim=1)  
            loss_adv = criterion(adv_out, labels[idx_batch[i]])

        # feature distillation
        if args.feat_distill:
            batch_mlp_emb = batch_mlp_emb[-1]
            batch_mlp_emb = model.encode_mlp4kd(batch_mlp_emb)
            batch_teacher_emb = teacher_emb[idx_batch[i]]
            mlp_sim_matrix = torch.mm(batch_mlp_emb, batch_mlp_emb.T)
            teacher_sim_matrix = torch.mm(batch_teacher_emb, batch_teacher_emb.T)
            loss_feature = torch.mean((teacher_sim_matrix - mlp_sim_matrix) ** 2)


        loss_label = criterion(out, labels[idx_batch[i]])
        loss = loss_label
        if args.adv:
            loss += 0.1 * loss_adv
        if args.feat_distill:
            loss += float(args.feat_distill_weight) * loss_feature
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        #if there is any parameter in optimizer requires_grad
        """back_flag=False
        for param in  optimizer.param_groups[0]['params']:
            if param.requires_grad:
                back_flag=True
                break
        if back_flag:
            loss.backward()"""
        try:
            loss.backward()
            optimizer.step()
        # if RuntimeError
        except RuntimeError as e:
            # element 0 of tensors does not require grad and does not have a grad_fn
            if "element 0 of tensors does not require grad and does not have a grad_fn" in str(e):
                pass
            else:
                raise e
            
        


    if args.verbose:
        print_str=f"Train loss: {total_loss / num_batches:.4f}"
        print(print_str)
    
    #time
    end_time = time.time()
    print("time for one epoch: ", end_time - start_time) if args.verbose else None
    args.time_cost_train_per_epoch.append(end_time - start_time)
    args.memory_cost_peak_train.append(torch.cuda.max_memory_allocated()/1024/1024)
    return total_loss / num_batches


def DE_regularization(model, feats, labels,graph, args, batch_size,  optimizer):
    # don't use batch, in this implementation, we use the whole dataset
    model.train()
    number_of_samples = feats.shape[0]

    #time
    start_time = time.time()
    #memory
    torch.cuda.reset_peak_memory_stats()
 
    device=labels.device
    total_loss = 0
    
    DE_ratio_by_layers=model.encoder.get_appoximated_DE(feats,graph) 
    teacher_related_layer_filter=[]
    for layer_num in range(len(DE_ratio_by_layers)):
        if model.encoder.transformation_types_pseudo[layer_num] in ["TeacherMLP", "TeacherGA"]:
            teacher_related_layer_filter.append(layer_num)
    DE_ratio_by_layers=[DE_ratio_by_layers[i] for i in teacher_related_layer_filter]
    # float or tensor
     # float or tensor
    args.DE_ratio_by_layers=[#x exp
    torch.exp(x).cpu().item() for x in DE_ratio_by_layers]
    args.DE_ratio_target_by_layers=[#x exp
    torch.exp(x).cpu().item() for x in model.encoder.DE_targets] 
    
    DE_targets=model.encoder.DE_targets

    assert len(DE_ratio_by_layers)==len(DE_targets)
    
    print("DE_targets",DE_targets) if args.verbose else None
    optimizer.zero_grad()
    loss_DE=0
    for layer_num in range(len(DE_ratio_by_layers)):
        #  MSE loss
        if args.sqrt_DER:
            power=0.5
        else:
            power=1
        if args.DE_log:
            temp_loss=(power*DE_ratio_by_layers[layer_num]-power*DE_targets[layer_num])**2
        else:
            temp_loss=(torch.exp(power*DE_ratio_by_layers[layer_num])-torch.exp(power*DE_targets[layer_num]))**2
        loss_DE+=temp_loss
        #loss_DE+=feats.sum()
            
    #loss_DE+=temp_loss
    loss_DE*=args.DE_regularization_rate
        
    #raise NotImplementedError
    total_loss += loss_DE.item()
    #if there is any parameter in optimizer requires_grad
    """back_flag=False
    for param in  optimizer.param_groups[0]['params']:
        if param.requires_grad:
            back_flag=True
            break
    if back_flag:
        loss.backward()"""
    try:
        loss_DE.backward()
 
        optimizer.step()
    # if RuntimeError
    except RuntimeError as e:
        # element 0 of tensors does not require grad and does not have a grad_fn
        if "element 0 of tensors does not require grad and does not have a grad_fn" in str(e):
            pass
        if "nan values" in str(e):
            # search across all gradients in optimizer
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            print("nan in grad")
                            print(p)
                            print(p.grad)
                            raise e
            
            raise e
        else:
            raise e
            
        


    if args.verbose:
        #print_str=f"\tSR loss: {total_loss / num_batches:.4f}"
        print_str=f"\tDE loss: {total_loss:.4f}"
        print(print_str)
    
    #time
    end_time = time.time()
    print("time for one epoch: ", end_time - start_time) if args.verbose else None
    args.time_cost_train_per_epoch.append(end_time - start_time)
    args.memory_cost_peak_train.append(torch.cuda.max_memory_allocated()/1024/1024)
    #return total_loss / num_batches
    return total_loss







def evaluate(model, data, feats, labels, criterion, evaluator, args,idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    #time
    start_time = time.time()
    model.eval()
    #memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if "GCN" in model.model_name or "GAT" in model.model_name or "APPNP" in model.model_name or "SimpleHGN" in model.model_name:
            
            if args.save_teacher_layer_info:
                emb_list, logits = model.inference(data, feats, save_teacher_layer_info=True)   
            else:
                emb_list, logits = model.inference(data, feats) 
        else:
            #MLP, sage
            if args.save_graph_aggregation:
                logits, emb_list = model.inference(data, feats,save_graph_aggregation=True)
            elif args.save_teacher_layer_info:
                logits, emb_list = model.inference(data, feats,save_teacher_layer_info=True)
            else:
                logits, emb_list = model.inference(data, feats)
        out = logits.log_softmax(dim=1)  
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    #time
    end_time = time.time()
    args.time_cost_eval_per_epoch.append(end_time - start_time)
    args.memory_cost_peak_eval.append(torch.cuda.max_memory_allocated()/1024/1024)
    return out, loss.item(), score, emb_list


def evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator,args, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    #time
    start_time = time.time()
    model.eval()
    device=labels.device
    #memory
    torch.cuda.reset_peak_memory_stats()
    number_of_samples = feats.shape[0]
    with torch.no_grad():
        num_batches = int(np.ceil(number_of_samples / batch_size))
        out_list = []
        #idx_batch = torch.randperm(number_of_samples)  # shuffle
        idx_batch = torch.arange(number_of_samples) # no shuffle
        if num_batches == 1:
            idx_batch = idx_batch.view(1, -1)
        else:
            idx_batch_list=[]
            for i in range(num_batches):
                if (i+1)*batch_size>number_of_samples:
                    idx_batch_list.append( idx_batch[i*batch_size:])
                else:
                    idx_batch_list.append( idx_batch[i*batch_size:(i+1)*batch_size])
            idx_batch=idx_batch_list
        for i in range(num_batches):
            
            batched_feats = feats[idx_batch[i]]
            #_, logits = model.inference(None, feats[batch_size * i: batch_size * (i + 1)])
            if args.save_student_layer_info:
                if i==0:
                    restart=True
                else:
                    restart=False
                _, logits = model.inference(None,batched_feats, save_student_layer_info=True,idx_b=idx_batch[i],total_num=number_of_samples,restart=restart)
            else:
                _, logits = model.inference(None, batched_feats)
            out = logits.log_softmax(dim=1) 
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])
        if args.verbose:
            print_str=f"Test loss: {loss.item():.4f}"
            for metric,v in score.items():
                print_str+=f"| {metric}: {v:.4f} "
            print(print_str)
    #time
    end_time = time.time()
    args.time_cost_eval_per_epoch.append(end_time - start_time)
    args.memory_cost_peak_eval.append(torch.cuda.max_memory_allocated()/1024/1024)
    return out_all, loss.item(), score


"""
2. Run teacher
"""


def run_transductive(
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
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    



    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:

        #
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
            
    else:
        g = g.to(device)
        data = g
        data_eval = g
    epoch=0
    #best_epoch, best_score_val, count = 0, 0, 0
    best_epoch, best_loss_val, count = 0,  float("inf"), 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(model, data, feats, labels, criterion, optimizer,args)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer,args
            )
        else:
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train, args)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator,args
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator,args
                )
                _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion, batch_size, evaluator,args
                )
            else:
                out, loss_train, score_train, emb_list = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator,args, idx_train
                    )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(
                #f"Ep {epoch:3d} | loss: {loss:.4f} | MacroF1_train: {score_train['macro_f1']:.4f} | MicroF1_train: {score_train['micro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MicroF1_val: {score_val['micro_f1']:.4f} | MacroF1_test: {score_test['macro_f1']:.4f} | MicroF1_test: {score_test['micro_f1']:.4f}"
                f"Ep {epoch:3d} | loss: {loss:.4f} | loss_train: {loss_train:.4f} | loss_val: {loss_val:.4f} | loss_test: {loss_test:.4f} | MacroF1_train: {score_train['macro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MacroF1_test: {score_test['macro_f1']:.4f}"
            )
            print(f"Ep {epoch:3d} | loss: {loss:.4f} | loss_train: {loss_train:.4f} | loss_val: {loss_val:.4f} | loss_test: {loss_test:.4f} | MacroF1_train: {score_train['macro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MacroF1_test: {score_test['macro_f1']:.4f}") if args.verbose else None
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            #if score_val["macro_f1"] >= best_score_val:
            if loss_val <= best_loss_val:
                best_epoch = epoch
                #best_score_val = score_val["macro_f1"]
                best_loss_val = loss_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if (count >= conf["patience"] and epoch>conf["max_epoch"]/2) or epoch == conf["max_epoch"]:
            break
    
    if epoch>0:
        model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator,args, idx_val
        )
        emb_list = None
    else:
        out, _, score_val, emb_list = evaluate(
                model, data_eval, feats, labels, criterion, evaluator,args, idx_val
            )
    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        #f"Best valid model at epoch: {best_epoch: 3d},  MacroF1_val: {best_score_val:.4f} | MacroF1_test: {score_test['macro_f1']:.4f} | MicroF1_test: {score_test['micro_f1']:.4f}"
        f"Best valid model at epoch: {best_epoch: 3d},  loss_val: {best_loss_val:.4f} | MacroF1_test: {score_test['macro_f1']:.4f} | MicroF1_test: {score_test['micro_f1']:.4f}"
    )
    return out, score_val, score_test, emb_list


def run_inductive(
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
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.DataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    #best_epoch, best_score_val, count = 0, 0, 0
    best_epoch, best_loss_val, count = 0, float("inf"), 0
    epoch=0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer,args
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer,args
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
                args,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator,args
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator,args
                )
                _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,args,
                )
                _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,args,
                )
            else:
                obs_out, loss_train, score_train, emb_list = evaluate(
                        model,
                        obs_data_eval,
                        obs_feats,
                        obs_labels,
                        criterion,
                        evaluator,args,
                        obs_idx_train,
                    )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                score_test_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                out, loss_test_ind, score_test_ind, emb_list = evaluate(
                        model, data_eval, feats, labels, criterion, evaluator,args, idx_test_ind
                    )
            logger.debug(
                #f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
                f"Ep {epoch:3d} | loss: {loss:.4f} | MacroF1_train: {score_train['macro_f1']:.4f} | MicroF1_train: {score_train['micro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MicroF1_val: {score_val['micro_f1']:.4f} | MacroF1_test_tran: {score_test_tran['macro_f1']:.4f} | MicroF1_test_tran: {score_test_tran['micro_f1']:.4f} | MacroF1_test_ind: {score_test_ind['macro_f1']:.4f} | MicroF1_test_ind: {score_test_ind['micro_f1']:.4f}"
            )
            print(f"Ep {epoch:3d} | loss: {loss:.4f} | MacroF1_train: {score_train['macro_f1']:.4f} | MicroF1_train: {score_train['micro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MicroF1_val: {score_val['micro_f1']:.4f} | MacroF1_test_tran: {score_test_tran['macro_f1']:.4f} | MicroF1_test_tran: {score_test_tran['micro_f1']:.4f} | MacroF1_test_ind: {score_test_ind['macro_f1']:.4f} | MicroF1_test_ind: {score_test_ind['micro_f1']:.4f}") if args.verbose else None
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            
            #if score_val["macro_f1"] >= best_score_val:
            if loss_val <= best_loss_val:
                best_epoch = epoch
                #best_score_val = score_val["macro_f1"]
                best_loss_val = loss_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if (count == conf["patience"] and epoch > conf["max_epoch"]/2) or epoch == conf["max_epoch"]:
            break
    if epoch>0:
        model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, criterion, batch_size, evaluator,args, obs_idx_val
        )
        out, _, score_test_ind = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator,args, idx_test_ind
        )

    else:
        obs_out, _, score_val, emb_list = evaluate(
                model,
                obs_data_eval,
                obs_feats,
                obs_labels,
                criterion,
                evaluator,args,
                obs_idx_val,
            )

        out, _, score_test_ind, emb_list = evaluate(
                model, data_eval, feats, labels, criterion, evaluator,args, idx_test_ind
            )

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        #f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
        #f"Best valid model at epoch: {best_epoch :3d}, macro_f1_val: {score_val['macro_f1'] :.4f}, score_test_tran: {score_test_tran['macro_f1'] :.4f}, score_test_ind: {score_test_ind['macro_f1'] :.4f}"
        f"Best valid model at epoch: {best_epoch :3d}, loss_val: {best_loss_val :.4f}, macro_f1_val: {score_val['macro_f1'] :.4f}, score_test_tran: {score_test_tran['macro_f1'] :.4f}, score_test_ind: {score_test_ind['macro_f1'] :.4f}"
    )
    if "MLP" in model.model_name:  # used in train_teacher with MLP as teacher model
        return out, score_val, score_test_tran, score_test_ind, None
    
    return out, score_val, score_test_tran, score_test_ind, emb_list


"""
3. Distill
"""


def distill_run_transductive(
        conf,
        model,
        feats,
        labels,
        out_t_all,
        out_emb_t_all,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        graph,
        args,
        from_learned_MLP_params=None,
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices
    # to device
    idx_l = idx_l.to(device)
    idx_t = idx_t.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    # sorted all indices
    idx_l = torch.sort(idx_l)[0]
    idx_t = torch.sort(idx_t)[0]
    idx_val = torch.sort(idx_val)[0]
    idx_test = torch.sort(idx_test)[0]
        

        
    
    feats = feats.to(device)
    feats_l=feats[idx_l]
    feats_t=feats[idx_t]
    feats_val=feats[idx_val]
    feats_test=feats[idx_test]

    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    """feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]"""
    labels_l = labels[idx_l]
    out_t = out_t_all[idx_t]
    labels_val = labels[idx_val]
    labels_test = labels[idx_test]

    out_emb_t = out_emb_t_all[idx_t]
    out_emb_l = out_emb_t_all[idx_l]

        
    #best_epoch, best_score_val, count = 0, 0, 0
    best_epoch, best_loss_val, count = 0,  float("inf"), 0
    for epoch in range(1, conf["max_epoch"] + 1):
        print(f"Epoch: {epoch}", end="\t") if args.verbose else None
        print(f"Task Loss: ", end="\t") if args.verbose else None

        loss_l = train_both_distillation_batch_adv(
            model, feats_l, labels_l, out_emb_l, args, batch_size, criterion_l, optimizer, lamb=1 - lamb,
        )
        print(f"Teacher Loss: ", end="\t") if args.verbose else None
        loss_t = train_both_distillation_batch_adv(
            model, feats_t, out_t, out_emb_t, args, batch_size, criterion_t, optimizer,lamb= lamb,
        )


        loss = loss_l + loss_t
        
        if args.DE_regularization:
            loss_DE=DE_regularization(model,feats,labels_l,graph,args,batch_size, optimizer)
            loss+=loss_DE
            

        if epoch % conf["eval_interval"] == 0:
            print(f"Train:", end="\t") if args.verbose else None
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator,args
            )
            print(f"Val:", end="\t") if args.verbose else None
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator,args
            )
            print(f"Test:", end="\t") if args.verbose else None
            _, loss_test, score_test = evaluate_mini_batch(
                model, feats_test, labels_test, criterion_l, batch_size, evaluator,args
            )

            logger.debug(
                #f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
                f"Ep {epoch:3d} | loss: {loss:.4f} | MacroF1_l: {score_l['macro_f1']:.4f} | MicroF1_l: {score_l['micro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MicroF1_val: {score_val['micro_f1']:.4f} | MacroF1_test: {score_test['macro_f1']:.4f} | MicroF1_test: {score_test['micro_f1']:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            
            #if score_val["macro_f1"] >= best_score_val:
            if loss_val <= best_loss_val:
                best_epoch = epoch
                #best_score_val = score_val["macro_f1"]
                best_loss_val = loss_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if (count == conf["patience"] and epoch > conf["max_epoch"]/2) or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator,args, idx_eval=idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        #f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
        #f"Best valid model at epoch: {best_epoch: 3d}, macro_f1_val: {score_val['macro_f1'] :.4f}, macro_f1_test: {score_test['macro_f1'] :.4f} micro_f1_test: {score_test['micro_f1'] :.4f}"
        f"Best valid model at epoch: {best_epoch: 3d}, loss_val: {best_loss_val :.4f}, macro_f1_val: {score_val['macro_f1'] :.4f}, micro_f1_val: {score_val['micro_f1'] :.4f}, macro_f1_test: {score_test['macro_f1'] :.4f}, micro_f1_test: {score_test['micro_f1'] :.4f}"
    )
    return out, score_val, score_test










def distill_run_inductive(
        conf,
        model,
        feats,
        labels,
        out_t_all,
        out_emb_t_all,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        args,
        from_learned_MLP_params=None,
        graph=None,  # used in training process
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """
    

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    out_emb_t = out_emb_t_all[obs_idx_t]
    out_emb_l = out_emb_t_all[obs_idx_l]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]


    #best_epoch, best_score_val, count = 0, 0, 0
    best_epoch, best_loss_val, count = 0,  float('inf'), 0
    for epoch in range(1, conf["max_epoch"] + 1):
        print(f"Epoch: {epoch}", end="\t") if args.verbose else None
        loss_l = train_both_distillation_batch_adv(
            model, feats_l, labels_l, out_emb_l, args, batch_size, criterion_l, optimizer, 1 - lamb, 
        )
        loss_t = train_both_distillation_batch_adv(
            model, feats_t, out_t, out_emb_t, args, batch_size, criterion_t, optimizer, lamb,
        )
        loss = loss_l + loss_t
            


        if args.DE_regularization:
            obs_feat=feats[idx_obs].to(device)
            obs_graph=dgl.node_subgraph(graph.to(device),idx_obs.to(device)).to(device)
            loss_DE=DE_regularization(model,obs_feat,labels_l,obs_graph,args,batch_size, optimizer)
            loss+=loss_DE
        

        if epoch % conf["eval_interval"] == 0:
            print(f"Label(Train) eval:", end="\t") if args.verbose else None
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator,args
            )
            print(f"Val eval:", end="\t") if args.verbose else None
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator,args
            )
            print(f"Test tran eval:", end="\t") if args.verbose else None
            _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
                args,
            )
            print(f"Test ind eval:", end="\t") if args.verbose else None
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
                args,
            )

            logger.debug(
                #f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
                f"Ep {epoch:3d} | loss: {loss:.4f} | MacroF1_l: {score_l['macro_f1']:.4f} | MicroF1_l: {score_l['micro_f1']:.4f} | MacroF1_val: {score_val['macro_f1']:.4f} | MicroF1_val: {score_val['micro_f1']:.4f} | MacroF1_test_tran: {score_test_tran['macro_f1']:.4f} | MicroF1_test_tran: {score_test_tran['micro_f1']:.4f} | MacroF1_test_ind: {score_test_ind['macro_f1']:.4f} | MicroF1_test_ind: {score_test_ind['micro_f1']:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            
            #if score_val["macro_f1"] >= best_score_val:
            if loss_val <= best_loss_val:
                best_epoch = epoch
                #best_score_val = score_val["macro_f1"]
                best_loss_val = loss_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if (count == conf["patience"] and epoch > conf["max_epoch"]/2) or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator,args, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator,args, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        #f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
        #f"Best valid model at epoch: {best_epoch: 3d} MacroF1_val: {score_val['macro_f1'] :.4f}, MicroF1_val: {score_val['micro_f1'] :.4f}, MacroF1_test_tran: {score_test_tran['macro_f1'] :.4f}, MicroF1_test_tran: {score_test_tran['micro_f1'] :.4f}, MacroF1_test_ind: {score_test_ind['macro_f1'] :.4f}, MicroF1_test_ind: {score_test_ind['micro_f1'] :.4f}"
        f"Best valid model at epoch: {best_epoch: 3d} loss_val: {best_loss_val :.4f}, MacroF1_test_tran: {score_test_tran['macro_f1'] :.4f}, MicroF1_test_tran: {score_test_tran['micro_f1'] :.4f}, MacroF1_test_ind: {score_test_ind['macro_f1'] :.4f}, MicroF1_test_ind: {score_test_ind['micro_f1'] :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
