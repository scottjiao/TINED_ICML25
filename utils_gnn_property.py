
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# dgl
import dgl
from dgl import DGLGraph

from dgl.nn import GraphConv,   APPNPConv, GATConv
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair,check_eq_shape






def compute_dirichlet_energy(graph,feature):
    """
    Compute the sqrt dirichlet energy of the graph
    G=(V,E)
    E(G,X)= (1/n)* \sum_{i\in V}\sum_{j| j is neighbor of i} ||x_i-x_j||^2
    return the dirichlet sqrt(E)
    ----------------------------------------
    input:
        graph: dgl graph n nodes, m edges
        feature: node feature n*d
    return:
        energy: scalar
    """
    n=graph.number_of_nodes()
    m=graph.number_of_edges()
    assert feature.shape[0]==n
    if len(feature.shape)==1:
        feature=feature.unsqueeze(1)
    elif len(feature.shape)==3:
        feature=feature.reshape(feature.shape[0],-1)


    #normalize feature by row
    # cut down the gradient of feature_norm
    
    feature_norm=torch.norm(feature,dim=1,p=2)
    feature_norm=feature_norm.unsqueeze(1).detach()
    # if nan or 0, replace with 1
    feature_norm[torch.isnan(feature_norm)]=1
    feature_norm[feature_norm==0]=1
    try:
        feature=feature/feature_norm
    except Exception as e:
        print(f"shape of feature: {feature.shape} | shape of feature_norm: {feature_norm.shape}")
        print(f"nodes number: {n} | edges number: {m}")
        raise e
    #if nan in feature
    #if torch.isnan(feature).any():
        #raise ValueError("feature is nan")
    #feature[torch.isnan(feature)]=0
    # get the edge weight
    with graph.local_scope():
        graph.ndata['x']=feature
        graph.apply_edges(fn.u_sub_v('x','x','edge_weight'))
        edge_weight=graph.edata['edge_weight']
        edge_weight=edge_weight**2

    # compute the energy
    energy=edge_weight.sum()/n


    # if is nan raise error
    #if torch.isnan(energy):
        #raise ValueError("energy is nan")

    return energy


def compute_mean_average_distance(graph,feature):
    
    """
    Compute the MAD of the graph
    G=(V,E)
    MAD(G,X)=  (1/n)* \sum_{i\in V}\sum_{j| j is neighbor of i} (1-<x_i,x_j>/(||x_i||*||x_j||))
    return the MAD
    ----------------------------------------
    input:
        graph: dgl graph n nodes, m edges
        feature: node feature n*d
    return:
        MAD: scalar
    """

    #normalize feature by row
    # cut down the gradient of feature_norm

    feature_norm=torch.norm(feature,dim=1,p=2)
    feature_norm=feature_norm.unsqueeze(1).detach()
    feature_norm[torch.isnan(feature_norm)]=1
    feature_norm[feature_norm==0]=1
    #
    feature=feature/feature_norm

    
    n=graph.number_of_nodes()
    m=graph.number_of_edges()
    assert feature.shape[0]==n
    if len(feature.shape)==1:
        feature=feature.unsqueeze(1)
    elif len(feature.shape)==3:
        feature=feature.reshape(feature.shape[0],-1)

    # get the edge weight: (1-<x_i,x_j>/(||x_i||*||x_j||))
    with graph.local_scope():
        graph.ndata['x']=feature
        graph.apply_edges(fn.u_mul_v('x','x','edge_weight'))
        edge_weight=graph.edata['edge_weight'].sum(dim=1)
        feature_norm=torch.norm(feature,dim=1,p=2)
        feature_norm=feature_norm.unsqueeze(1)
        # 1/(||x_i||*||x_j||)
        graph.ndata['feature_norm']=1/feature_norm
        graph.apply_edges(fn.u_mul_v('feature_norm','feature_norm','edge_weight2'))
        edge_weight2=graph.edata['edge_weight2']
        edge_weight2=edge_weight2.squeeze(-1)
        #print(edge_weight.shape,edge_weight2.shape)
        edge_weight=torch.mul(edge_weight,edge_weight2)
        edge_weight=1-edge_weight
    # compute the MAD
    MAD=edge_weight.sum()/n

    return MAD



# test samples

if __name__=="__main__":
    # test compute on MAD
    # 1. test on a complete graph
    # 2. test on a star graph
    # 3. test on a path graph

    # 1. test on a complete graph with 3 nodes
    g=dgl.graph(([0,1,2],[1,2,0]))
    # 1.1 id vector as feature
    feature=torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.float)
    MAD=compute_mean_average_distance(g,feature)
    print(f"MAD of a complete graph with 3 nodes in id feature: {MAD}")
    # 1.2 simply constant vector as feature
    feature=torch.tensor([[1,1,0],[1,0,1],[0,1,1]],dtype=torch.float)
    MAD=compute_mean_average_distance(g,feature)
    print(f"MAD of a complete graph with 3 nodes in constant feature: {MAD}")

    # 2. test on a star graph with 3 nodes
    g=dgl.graph(([0,0],[1,2]))
    # 2.1 id vector as feature
    feature=torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.float)
    MAD=compute_mean_average_distance(g,feature)
    print(f"MAD of a star graph with 3 nodes in id feature: {MAD}")
    # 2.2 simply constant vector as feature
    feature=torch.tensor([[1,1,0],[1,0,1],[0,1,1]],dtype=torch.float)
    MAD=compute_mean_average_distance(g,feature)
    print(f"MAD of a star graph with 3 nodes in constant feature: {MAD}")





