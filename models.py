import torch
import torch as th
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
#from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv
from dgl.nn import   APPNPConv
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair,check_eq_shape
from utils_gnn_property import compute_dirichlet_energy
import numpy as np

#Tuple, List, Optional, Union, Any, Callable, Dict, Tuple, List, Optional, Union, Any, Callable, Dict
from typing import Tuple, List, Optional, Union, Any, Callable, Dict

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Optional
from torch_geometric.nn.models import LabelPropagation

from pathlib import Path
# pylint: enable=W0235
class GATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False,save_teacher_layer_info=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            #GAT has a fixed order: first linear transformation, then graph aggregation
            trans_order=["MLP","GA"]
            in_hs=[h_src]
            out_hs=[feat_src]
            #h1=feat_src
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            in_hs.append(feat_src)
            out_hs.append(rst)
            #h2=rst
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                raise NotImplementedError
                return rst, graph.edata["a"]
            else:
                
                if save_teacher_layer_info:
                    return rst,in_hs,out_hs,trans_order
                    #h,in_hs,out_hs,trans_order
                else:
                    return rst
            
    def get_MLP_layers(self):
        # get the mlp part from gnn layers (sage layers with 'gcn' aggregation)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            mlp_layers.append(layer.weight)
            mlp_layers.append(layer.bias)


        return mlp_layers

    def get_teacher_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.teacher_layer_info



# pylint: disable=W0235
class GraphConv(nn.Module):
    r"""Graph convolutional layer from `Semi-Supervised Classification with Graph Convolutional
    Networks <https://arxiv.org/abs/1609.02907>`__

    Mathematically it is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ji}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`),
    and :math:`\sigma` is an activation function.

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{e_{ji}}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    This is NOT equivalent to the weighted graph convolutional network formulation in the paper.

    To customize the normalization term :math:`c_{ji}`, one can first set ``norm='none'`` for
    the model, and send the pre-normalized :math:`e_{ji}` to the forward computation. We provide
    :class:`~dgl.nn.pytorch.EdgeWeightNorm` to normalize scalar edge weight following the GCN paper.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer.  Can be one of the following values:

        * ``right``, to divide the aggregated messages by each node's in-degrees,
          which is equivalent to averaging the received messages.

        * ``none``, where no normalization is applied.

        * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}` above, equivalent
          to symmetric normalization.

        * ``left``, to divide the messages sent out from each node by its out-degrees,
          equivalent to random walk normalization.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GraphConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[ 1.3326, -0.2797],
            [ 1.4673, -0.3080],
            [ 1.3326, -0.2797],
            [ 1.6871, -0.3541],
            [ 1.7711, -0.3717],
            [ 1.0375, -0.2178]], grad_fn=<AddBackward0>)
    >>> # allow_zero_in_degree example
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[-0.2473, -0.4631],
            [-0.3497, -0.6549],
            [-0.3497, -0.6549],
            [-0.4221, -0.7905],
            [-0.3497, -0.6549],
            [ 0.0000,  0.0000]], grad_fn=<AddBackward0>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('_U', '_E', '_V') : (u, v)})
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 5)
    >>> conv = GraphConv(5, 2, norm='both', weight=True, bias=True)
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[-0.2994,  0.6106],
            [-0.4482,  0.5540],
            [-0.5287,  0.8235],
            [-0.2994,  0.6106]], grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GraphConv, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None,save_teacher_layer_info=False):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                trans_order=["MLP","GA"]
                # hidden_feature_1
                in_hs=[feat_src]
                h1=th.matmul(feat_src, weight)
                out_hs=[h1]
                if weight is not None:
                    #feat_src = th.matmul(feat_src, weight)
                    feat_src=h1
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                trans_order=["GA","MLP"]
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                """if weight is not None:
                    rst = th.matmul(rst, weight)"""  # place it behind the graph normalization

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm


            if self._in_feats > self._out_feats:
                in_hs.append(feat_src)
                out_hs.append(rst)
                h2=rst
            else:
                in_hs=[feat_src,rst]
                h1=rst
                out_hs=[rst]
                if weight is not None:
                    rst = th.matmul(rst, weight)
                out_hs.append(rst)
                h2=rst

            if self.bias is not None:
                rst = rst + self.bias


            if self._activation is not None:
                rst = self._activation(rst)

            if save_teacher_layer_info:
                #return rst,h1,h2,trans_order
                return rst, in_hs, out_hs,trans_order
            else:
                return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None,save_graph_aggregation=False,save_teacher_layer_info=False):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats
            if lin_before_mp:
                trans_order=["MLP","GA"]
            else:
                trans_order=["GA","MLP"]

            
            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  
                    if lin_before_mp:
                        h1 = self.fc_neigh(feat_dst)
                        in_hs=[feat_dst.detach().cpu(),h1.detach().cpu()]
                        out_hs=[h1.detach().cpu()]
                        features_before_aggr=h1 
                    else:
                        features_before_aggr=feat_dst
                        
                    graph.dstdata["h"] = (
                        features_before_aggr
                    )
                    
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                            : graph.num_dst_nodes()
                        ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                features_after_aggr=h_neigh
                if not lin_before_mp:
                    h1=h_neigh
                    h_neigh = self.fc_neigh(h_neigh)
                    in_hs=[feat_dst.detach().cpu(),h1.detach().cpu()]
                    out_hs=[h1.detach().cpu(),h_neigh.detach().cpu()]
                
                    h2=h_neigh
                else:
                    h2=h_neigh
                    out_hs.append(h_neigh.detach().cpu())
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            
            if save_graph_aggregation:
                return rst,features_before_aggr,features_after_aggr
            elif save_teacher_layer_info:
                #return rst,h1,h2,trans_order
                return rst,in_hs,out_hs,trans_order
            else:
                return rst


class MLP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            position_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
            graph=None,
            byte_idx_train=None,
            labels_one_hot=None,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim + position_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim + position_dim, hidden_dim))

            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.teacher_feature_encoder = nn.Linear(hidden_dim, hidden_dim)
            self.mlp_feature_encoder = nn.Linear(hidden_dim, hidden_dim)
            self.mlp_feature_encoder_2 = nn.Linear(hidden_dim, hidden_dim)
            


    def forward(self, feats,save_student_layer_info=False,idx_b=None,total_num=None,restart=False):
        
        if save_student_layer_info:
            if restart:
                student_layer_info=[]
            else:
                student_layer_info=self.student_layer_info
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            if save_student_layer_info:
                h_old=h
            h = layer(h)
            if save_student_layer_info:
                if restart:
                    student_layer_info_layer={}
                    student_layer_info_layer["transformation_type"]= "MLP"
                    student_layer_info_layer["MLP"]=layer.weight
                    student_layer_info_layer["feature_matrix_in"]= torch.zeros(total_num,h_old.shape[1]).to(h_old.device)
                    student_layer_info_layer["feature_matrix_out"]= torch.zeros(total_num,h.shape[1]).to(h_old.device)
                    student_layer_info_layer["feature_matrix_in"][idx_b]=h_old
                    student_layer_info_layer["feature_matrix_out"][idx_b]=h
                    student_layer_info.append(student_layer_info_layer)
                else:
                    student_layer_info[l]["feature_matrix_in"][idx_b]=h_old
                    student_layer_info[l]["feature_matrix_out"][idx_b]=h

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        if save_student_layer_info:
            self.student_layer_info=student_layer_info
        return h_list, h

    def encode_teacher4kd(self, teacher_feat):
        return self.dropout(F.relu(self.teacher_feature_encoder(teacher_feat)))

    def encode_mlp4kd(self, mlp_feat):
        return self.dropout(F.relu(self.mlp_feature_encoder(mlp_feat)))
    
    def encode_mlp4attn_distill(self, mlp_feat,layer_num):
        return self.dropout(F.relu(self.mlp_feature_encoder_attn_distill[layer_num](mlp_feat)))
    
    def encode_mlp4lp(self, mlp_feat,layer_num):
        return self.dropout(F.relu(self.mlp_feature_encoder_lp[layer_num](mlp_feat)))

    def encode_mlp4etype_pred(self, mlp_feat,layer_num):
        return self.dropout(F.relu(self.mlp_feature_encoder_etype_pred[layer_num](mlp_feat)))
    
    def get_MLP_layers(self):
        # get the mlp part from gnn layers (sage layers with 'gcn' aggregation)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            layer_obj=layer
            mlp_layers.append(layer_obj.weight)
            mlp_layers.append(layer_obj.bias)


        return mlp_layers

    def get_student_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.student_layer_info


"""
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
"""


#nn.linear with skip connection X=X+nn.linear(X)
class skip_connection_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(skip_connection_Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # weight is pointer to self.linear.weight
        self.weight = self.linear.weight
        # bias is pointer to self.linear.bias
        self.bias = self.linear.bias
    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return x + self.linear(x)



class MLP_from_sequence_of_layers(nn.Module):
    def __init__(
            self,
            num_layers,
            shape_sequence_of_teacher_layers,
            shape_learned_graph_aggregation_layers,
            input_dim,
            position_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
            graph=None,
            byte_idx_train=None,
            labels_one_hot=None,
            from_MLP_mode="same_as_teacher",
            args=None
    ):
    # shape_sequence_of_teacher_layers: [[m1shape1,m2shape2],[m2shape1,m2shape2],...]
    # the constructed MLP will be [input_dim+position_dim,m1shape1],[m1shape1,m1shape2],
    #                               [m1shape2,m2shape1],[m2shape1,m2shape2],
    #                               [m2shape2,m3shape1],[m3shape1,m3shape2],...
    #                               [m(n)shape2,output_dim]]
    # the teacher layers are all fixed: [m1shape1,m1shape2],[m2shape1,m2shape2],...,[m(n)shape1,m(n)shape2]

    
        super(MLP_from_sequence_of_layers, self).__init__()
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        if norm_type== "batch":
            norm_func = nn.BatchNorm1d 
        elif norm_type == "layer":
            norm_func = nn.LayerNorm
        elif norm_type == "none":
            norm_func = nn.Identity
        else:
            raise NotImplementedError

        self.from_MLP_mode=from_MLP_mode
        self.learned_MLP_layers=nn.ModuleList()
        self.graph_aggregation_layers=nn.ModuleList()
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim
        self.shape_sequence_of_teacher_layers=shape_sequence_of_teacher_layers
        self.shape_learned_graph_aggregation_layers=shape_learned_graph_aggregation_layers
        
        self.transformation_types_pseudo=[] # only learned MLP layers regarded as "MLP", other MLP layers are "GA"
        self.load_disabled=args.load_disabled
        self.DE_sampling_ratio=args.DE_sampling_ratio
        self.args=args
        
        self.input_dim=input_dim
        self.position_dim=position_dim
        
            
        
        if self.from_MLP_mode=="same_as_teacher":

            """
            The architecture is the same as above "learned_graph_aggregation", but don't load the parameters of graph aggregation layers
            """

            assert len(self.shape_learned_graph_aggregation_layers)==len(self.shape_sequence_of_teacher_layers)
            # 1 means feature transformation before graph aggregation, 0 means feature transformation after graph aggregation
            order_list=[]
            for i in range(len(self.shape_sequence_of_teacher_layers)):
                if self.shape_sequence_of_teacher_layers[i][0]<=self.shape_sequence_of_teacher_layers[i][1]:
                    order_list.append(0)
                else:
                    order_list.append(1)
            # now we have the order list, we can construct the distilled MLP
            if len(shape_sequence_of_teacher_layers) == 0:
                raise NotImplementedError
            for i in range(len(shape_sequence_of_teacher_layers)):

                if order_list[i]==1:
                    # feature transformation before graph aggregation
                    if i==0 and input_dim + position_dim!=shape_sequence_of_teacher_layers[0][0]:  #position_dim>0
                        learned_MLP_layer=nn.Linear(input_dim + position_dim, shape_sequence_of_teacher_layers[i][1])
                    else:
                        learned_MLP_layer=nn.Linear(shape_sequence_of_teacher_layers[i][0], shape_sequence_of_teacher_layers[i][1])
                    self.layers.append(learned_MLP_layer)
                    self.transformation_types_pseudo.append("TeacherMLP")
                    self.learned_MLP_layers.append(learned_MLP_layer)
                    #norm
                    self.norms.append(norm_func(shape_sequence_of_teacher_layers[i][1]))
                    # graph aggregation
                    graph_aggregation_layer=nn.Linear(self.shape_learned_graph_aggregation_layers[i][0], self.shape_learned_graph_aggregation_layers[i][1])
                    
                    self.layers.append(graph_aggregation_layer)
                    self.transformation_types_pseudo.append("TeacherGA")
                    self.graph_aggregation_layers.append(graph_aggregation_layer)
                    self.norms.append(norm_func(self.shape_learned_graph_aggregation_layers[i][1]))
                else:
                    if i==0 and input_dim + position_dim!=shape_sequence_of_teacher_layers[0][0]:
                        graph_aggregation_layer=nn.Linear(input_dim + position_dim, self.shape_learned_graph_aggregation_layers[i][1])
                    else:
                    
                        graph_aggregation_layer=nn.Linear(self.shape_learned_graph_aggregation_layers[i][0], self.shape_learned_graph_aggregation_layers[i][1])
                    # graph aggregation
                    
                    self.layers.append(graph_aggregation_layer)
                    self.transformation_types_pseudo.append("TeacherGA")
                    self.graph_aggregation_layers.append(graph_aggregation_layer)
                    self.norms.append(norm_func(self.shape_learned_graph_aggregation_layers[i][1]))
                    # feature transformation after graph aggregation
                    learned_MLP_layer=nn.Linear(shape_sequence_of_teacher_layers[i][0], shape_sequence_of_teacher_layers[i][1])
                    
                    self.layers.append(learned_MLP_layer)
                    self.transformation_types_pseudo.append("TeacherMLP")
                    self.learned_MLP_layers.append(learned_MLP_layer)
                    #norm
                    self.norms.append(norm_func(shape_sequence_of_teacher_layers[i][1]))
            if shape_sequence_of_teacher_layers[-1][1]!=output_dim:
                # dont tolerate the output dim is different from the teacher model
                raise Exception("The output dim of the distilled MLP is different from the teacher model")
            self.teacher_feature_encoder = nn.Linear(shape_sequence_of_teacher_layers[-1][1], shape_sequence_of_teacher_layers[-1][1])
            self.mlp_feature_encoder = nn.Linear(shape_sequence_of_teacher_layers[-1][1], shape_sequence_of_teacher_layers[-1][1])
        
        
        elif self.from_MLP_mode=="same_as_teacher_appnp":
            # APPNP decouples the feature transformation and graph aggregation
            # the first several layers are feature transformation, the last layer is graph aggregation
            # directly use the shape_sequence_of_teacher_layers as the MLP
            # then add the graph aggregation layer
            if len(shape_sequence_of_teacher_layers) == 0:
                raise NotImplementedError
            # directly use the shape_sequence_of_teacher_layers as the MLP
            for i in range(len(shape_sequence_of_teacher_layers)):
                #[m(i)shape1,m(i)shape2] # fixed
                if i==0 and input_dim + position_dim!=shape_sequence_of_teacher_layers[0][0]:  #position_dim>0
                    learned_MLP_layer=nn.Linear(input_dim + position_dim, shape_sequence_of_teacher_layers[i][1])
                else:
                    learned_MLP_layer=nn.Linear(shape_sequence_of_teacher_layers[i][0], shape_sequence_of_teacher_layers[i][1])
                self.layers.append(learned_MLP_layer)
                self.transformation_types_pseudo.append("TeacherMLP")
                self.learned_MLP_layers.append(learned_MLP_layer)
                #norm
                self.norms.append(norm_func(shape_sequence_of_teacher_layers[i][1]))
            # add the graph aggregation layer
            self.layers.append(nn.Linear(shape_sequence_of_teacher_layers[-1][1], output_dim))
            self.transformation_types_pseudo.append("TeacherGA")
            self.norms.append(norm_func(output_dim))
            self.teacher_feature_encoder = nn.Linear(shape_sequence_of_teacher_layers[-1][1], shape_sequence_of_teacher_layers[-1][1])
            self.mlp_feature_encoder = nn.Linear(shape_sequence_of_teacher_layers[-1][1], shape_sequence_of_teacher_layers[-1][1])
        
        else:
            raise NotImplementedError
        


        
        if args.DE_regularization:
            self.DE_targets=[]
            # load the DE targets of the teacher models
            if args.DE_mode=="same_as_teacher":
                self.DE_targets=self.load_teacher_DE_targets(args) # sequence of DE targets of teacher models
            elif args.DE_mode=="target":
                raise NotImplementedError
            elif args.DE_mode=="ones": # for dynamic DE regularization
                for i in range(len( self.layers)):
                    self.DE_targets.append(1.0)
            

        
    def initialize(self,GA_init_type):
        # initialize the learned graph aggregation layers
        if GA_init_type=="random":  #   GA_init_type:["random","identity"]
            pass
        elif GA_init_type=="identity":
            # initialize the learned graph aggregation layers as identity K*K
            for i in range(len(self.graph_aggregation_layers)):
                self.graph_aggregation_layers[i].weight.data=torch.eye(self.graph_aggregation_layers[i].weight.shape[0],self.graph_aggregation_layers[i].weight.shape[1])
                self.graph_aggregation_layers[i].bias.data=torch.zeros(self.graph_aggregation_layers[i].bias.shape[0])

                

    def forward(self, feats,save_student_layer_info=False,idx_b=None,total_num=None,restart=False):
        if save_student_layer_info:
            if restart:
                student_layer_info=[]
            else:
                student_layer_info=self.student_layer_info
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            if save_student_layer_info:
                h_old=h
            h = layer(h)
            if save_student_layer_info:
                if restart:
                    student_layer_info_layer={}
                    student_layer_info_layer["transformation_type"]= "MLP"
                    student_layer_info_layer["MLP"]=layer.weight
                    student_layer_info_layer["feature_matrix_in"]= torch.zeros(total_num,h_old.shape[1]).to(h_old.device)
                    student_layer_info_layer["feature_matrix_out"]= torch.zeros(total_num,h.shape[1]).to(h_old.device)
                    #print(student_layer_info_layer["feature_matrix_in"].shape, student_layer_info_layer["feature_matrix_in"][idx_b].shape)
                    student_layer_info_layer["feature_matrix_in"][idx_b]=h_old
                    student_layer_info_layer["feature_matrix_out"][idx_b]=h
                    student_layer_info.append(student_layer_info_layer)
                else:
                    student_layer_info[l]["feature_matrix_in"][idx_b]=h_old
                    student_layer_info[l]["feature_matrix_out"][idx_b]=h

            if l != len(self.layers) - 1:
                h_list.append(h)
                h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        if save_student_layer_info:
            self.student_layer_info=student_layer_info
        
        return h_list, h
    def get_appoximated_spectrum(self,feats):
        h = feats
        h_list = []
        max_spectrum_by_layers,min_spectrum_by_layers = [],[]
        for l, layer in enumerate(self.layers):
            h_temp=h
            h = layer(h)

            # get the derivable and approximate spectrum of the layer
            norm_in=torch.norm(h_temp,dim=1)
            norm_out=torch.norm(h,dim=1)
            ratios=norm_out/(norm_in+1e-8)
            # filter out nan and inf
            ratios=ratios[torch.isfinite(ratios)]

            max_spectrum_by_layers.append(torch.max(ratios))
            min_spectrum_by_layers.append(torch.min(ratios))

            if l != len(self.layers) - 1:
                h_list.append(h)
                h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        return max_spectrum_by_layers,min_spectrum_by_layers
    
    def get_appoximated_DE(self,feats,graph):
        # DE: Dirichlet energy
        h = feats
        h_list = []
        DE_ratio_by_layers = [] 
        if self.DE_sampling_ratio!=1.0:
            graph_=graph.edge_subgraph(np.random.choice(graph.number_of_edges(),int(self.DE_sampling_ratio*graph.number_of_edges()),replace=False),relabel_nodes=False)
            # in and out
            isolated_nodes_bool=(graph_.in_degrees()==0) & (graph_.out_degrees()==0)
            non_isolated_nodes_bool=~isolated_nodes_bool
            isolated_nodes=torch.where(isolated_nodes_bool)[0]
            non_isolated_nodes=torch.where(non_isolated_nodes_bool)[0]
            graph_.remove_nodes(isolated_nodes)
            # get non-isolated nodes features
            h=h[non_isolated_nodes]
        else:
            graph_=graph
        for l, layer in enumerate(self.layers):
            h_temp1=h
            h = layer(h)
            h_temp2=h

            # get the derivable and approximate spectrum of the layer
            
            
                
            DE_in= compute_dirichlet_energy(graph_,h_temp1)
            DE_out= compute_dirichlet_energy(graph_,h_temp2)
            


            #ratio=DE_out/(DE_in+1e-8)
            #logarithm
            """if DE_in==0:
                raise Exception("DE_in is 0")
            if DE_out==0:
                raise Exception("DE_out is 0")"""
            if DE_in==0 or DE_out==0:
                ratio=torch.tensor(1.0,dtype=torch.float32)
            else:
            
                ratio=torch.log(DE_out)-torch.log(DE_in)
            # filter out nan and inf
            #ratios=ratios[torch.isfinite(ratio)]
            # if nan
            DE_ratio_by_layers.append( ratio)
            

            if l != len(self.layers) - 1:
                h_list.append(h)
                h = self.norms[l](h)
                h = F.relu(h)
                # if all the elements in h are 0, then DE is 0
                h = self.dropout(h)
        return DE_ratio_by_layers

    def load_teacher_DE_targets(self,args):
        if args.exp_setting == "tran":
            out_t_dir = Path.cwd().joinpath(
                args.out_t_path,
                "transductive",
                args.dataset,
                args.teacher,
                f"seed_{args.seed}",
            )
        elif args.exp_setting == "ind":
            out_t_dir = Path.cwd().joinpath(
                args.out_t_path,
                "inductive",
                f"split_rate_{args.split_rate}",
                args.dataset,
                args.teacher,
                f"seed_{args.seed}",
            )
        # load teacher information about DE
        teacher_DE_targets=torch.load(out_t_dir.joinpath("DE_targets.pt")) # list of float numbers
        # to tensor
        teacher_DE_targets=torch.tensor(teacher_DE_targets,dtype=torch.float32)
        # if contain null or nan, inf, then replace them with 1.0
        for i in range(len(teacher_DE_targets)):
            if teacher_DE_targets[i] is None or not torch.isfinite(teacher_DE_targets[i]):
                teacher_DE_targets[i]=1.0
        # logarithm
        teacher_DE_targets=torch.log(teacher_DE_targets)
        return teacher_DE_targets
    def encode_teacher4kd(self, teacher_feat):
        return self.dropout(F.relu(self.teacher_feature_encoder(teacher_feat)))

    def encode_mlp4kd(self, mlp_feat):
        try:
            return self.dropout(F.relu(self.mlp_feature_encoder(mlp_feat)))
        except:
            print(mlp_feat.shape)
            print(self.hidden_dim)
            print(self.mlp_feature_encoder.weight.shape)
            print(self.mlp_feature_encoder.bias.shape)
            raise NotImplementedError
    
    def load_learned_params(self,from_learned_MLP_params):
        if self.load_disabled:
            return None
        learned_MLP_layers=from_learned_MLP_params['learned_MLP_layers']# list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....] # loaded from the teacher model
        # NOTE: variable "self.learned_MLP_layers" is student's own MLP layers needed to be loaded from the teacher model's variable "learned_MLP_layers"
        if "learned_graph_aggregation" or "same_as_teacher" in self.from_MLP_mode:
            learned_graph_aggregation=from_learned_MLP_params['learned_graph_aggregation']# list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        #print shapes
        print("learned_MLP_layers:")
        for i in range(len(learned_MLP_layers)):
            print(learned_MLP_layers[i].shape)
        print("self.layers:")
        for i in range(len(self.layers)):
            print(self.layers[i].weight.shape)
        if self.from_MLP_mode=="gap":
            for i in range(len(self.layers)):
                if i%2==1:
                    #assert shape
                    print(f"load layer {i} with shape  {self.layers[i].weight.shape} to {learned_MLP_layers[i-1].shape }")
                    assert self.layers[i].weight.shape==learned_MLP_layers[i-1].shape 
                    self.layers[i].weight.data=learned_MLP_layers[i-1]
                    self.layers[i].bias.data=learned_MLP_layers[i]
                    ## and set fixed
                    #for param in self.layers[i].parameters():
                    #    param.requires_grad = False
                else:
                    # intermediate layer is to learn 
                    pass 
        elif self.from_MLP_mode=="simple":
            for i in range(len(self.layers)):
                #assert shape
                print(f"load layer {i} with shape  {self.layers[i].weight.shape} to {learned_MLP_layers[2*i].shape }")
                assert self.layers[i].weight.shape==learned_MLP_layers[2*i].shape 
                self.layers[i].weight.data=learned_MLP_layers[2*i]
                self.layers[i].bias.data=learned_MLP_layers[2*i+1]
                ## and set fixed
                #for param in self.layers[i].parameters():
                    #param.requires_grad = False
        elif self.from_MLP_mode=="learned_graph_aggregation":
            for i,l in enumerate(self.learned_MLP_layers):
                #assert shape
                if i==0 and self.input_dim + self.position_dim!=self.shape_sequence_of_teacher_layers[0][0] and self.shape_sequence_of_teacher_layers[i][0]>self.shape_sequence_of_teacher_layers[i][1]:
                    print(f"Addtional parameter in first layer! Only the first part of the weight and bias of the first layer of student is loaded from the learned MLP")
                    # teacher.l.weight matrix: self.input_dim * self.shape_sequence_of_teacher_layers[0][0]
                    # student.l.weight matrix: self.input_dim + self.position_dim * self.shape_sequence_of_teacher_layers[0][0]
                    l.weight.data[:,:  self.input_dim]=learned_MLP_layers[2*i]
                    l.bias.data[:  self.input_dim]=learned_MLP_layers[2*i+1]
                else:
                    print(f"load layer {i} with shape  {l.weight.shape} to {learned_MLP_layers[2*i].shape }")
                    assert l.weight.shape==learned_MLP_layers[2*i].shape 
                    l.weight.data=learned_MLP_layers[2*i]
                    l.bias.data=learned_MLP_layers[2*i+1]
            # load graph aggregation layers
            for i,l in enumerate(self.graph_aggregation_layers):
                #assert shape
                print(f"load layer {i} with shape  {l.weight.shape} to {learned_graph_aggregation[2*i].shape }")
                assert l.weight.shape==learned_graph_aggregation[2*i].shape
                l.weight.data=learned_graph_aggregation[2*i]
                l.bias.data=learned_graph_aggregation[2*i+1]
                
        else:
            
            for i,l in enumerate(self.learned_MLP_layers):
                additional_param=False
                if i==0 and self.input_dim + self.position_dim!=self.shape_sequence_of_teacher_layers[0][0]:
                    if self.shape_sequence_of_teacher_layers[i][0]>self.shape_sequence_of_teacher_layers[i][1]:
                        additional_param=True
                    if self.from_MLP_mode=="the_same_as_teacher_appnp":
                        additional_param=True



                #assert shape
                if additional_param:
                    print(f"Addtional parameter in first layer! Only the first part of the weight and bias of the first layer of student is loaded from the learned MLP")
                    # teacher.l.weight matrix: self.input_dim * self.shape_sequence_of_teacher_layers[0][0]
                    # student.l.weight matrix: self.input_dim + self.position_dim * self.shape_sequence_of_teacher_layers[0][0]
                    l.weight.data[:,:  self.input_dim]=learned_MLP_layers[2*i]
                    l.bias.data[:  self.input_dim]=learned_MLP_layers[2*i+1]
                else:
                    print(f"load layer {i} with shape  {l.weight.shape} to {learned_MLP_layers[2*i].shape }")
                    assert l.weight.shape==learned_MLP_layers[2*i].shape 
                    l.weight.data=learned_MLP_layers[2*i]
                    l.bias.data=learned_MLP_layers[2*i+1]

   
    def get_MLP_layers(self):
        # get the mlp part from gnn layers (sage layers with 'gcn' aggregation)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            layer_obj=layer
            mlp_layers.append(layer_obj.weight)
            mlp_layers.append(layer_obj.bias)


        return mlp_layers

    def get_student_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.student_layer_info





class SAGE(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, "gcn"))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, "gcn"))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, "gcn"))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(SAGEConv(hidden_dim, output_dim, "gcn"))

    def forward(self, blocks, feats):
        h = feats
        h_list = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            
            h= layer(block, (h, h_dst))
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_list, h

    def inference(self, dataloader, feats,save_graph_aggregation=False,save_teacher_layer_info=False):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        dataloader : The entire graph loaded in blocks with full neighbors for each node.
        feats : The input feats of entire node set.
        """
        device = feats.device
        emb_list = []
        if save_graph_aggregation:
            features_before_aggr=[]
            features_after_aggr=[]
        if save_teacher_layer_info:
            teacher_layer_info= []
        # initial feature
        

        for l, layer in enumerate(self.layers):
            # add-on
            hidden_emb = torch.zeros(
                feats.shape[0],
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)

            y = torch.zeros(
                feats.shape[0],
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)

            if save_graph_aggregation:
                features_before_aggr_l = torch.zeros(feats.shape[0],self.hidden_dim if l != self.num_layers - 1 else self.output_dim,).to(device)
                features_after_aggr_l = torch.zeros(feats.shape[0],self.hidden_dim if l != self.num_layers - 1 else self.output_dim,).to(device)
            #if save_teacher_layer_info:
                
                #h1_l = torch.zeros(feats.shape[0],self.hidden_dim if l != self.num_layers - 1 else self.output_dim,).to(device)
                #h2_l = torch.zeros(feats.shape[0],self.hidden_dim if l != self.num_layers - 1 else self.output_dim,).to(device)

            count = 0
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = feats[input_nodes]
                h_dst = h[: block.num_dst_nodes()]

                if save_graph_aggregation:
                    h,features_before_aggr_b,features_after_aggr_b = layer(block, (h, h_dst),save_graph_aggregation=save_graph_aggregation)
                    features_before_aggr_l[output_nodes] = features_before_aggr_b
                    features_after_aggr_l[output_nodes] = features_after_aggr_b
                elif save_teacher_layer_info:
                    #h,h1,h2,trans_order = layer(block, (h, h_dst),save_teacher_layer_info=save_teacher_layer_info)
                    h,in_hs,out_hs,trans_order = layer(block, (h, h_dst),save_teacher_layer_info=save_teacher_layer_info)
                    if count==0:
                        in_h1=torch.zeros(feats.shape[0],in_hs[0].shape[1]).cpu()
                        in_h2=torch.zeros(feats.shape[0],in_hs[1].shape[1]).cpu()
                        out_h1=torch.zeros(feats.shape[0],out_hs[0].shape[1]).cpu()
                        out_h2=torch.zeros(feats.shape[0],out_hs[1].shape[1]).cpu()
                    in_h1[output_nodes]=in_hs[0]
                    in_h2[output_nodes]=in_hs[1]
                    out_h1[output_nodes]=out_hs[0]
                    out_h2[output_nodes]=out_hs[1]
                    
                    
                else:
                    h = layer(block, (h, h_dst))

                if l != self.num_layers - 1:
                    hidden_emb[output_nodes] = h
                    if self.norm_type != "none":
                        h = self.norms[l](h)
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

                count += 1

            feats = y
            if save_graph_aggregation:
                features_before_aggr.append(features_before_aggr_l)
                features_after_aggr.append(features_after_aggr_l)
            if save_teacher_layer_info:

                
                teacher_layer_info_l1={}
                teacher_layer_info_l1["transformation_type"]=trans_order[0]
                if trans_order[0]=="MLP":
                    teacher_layer_info_l1["MLP"]=layer.fc_neigh.weight
                else:
                    teacher_layer_info_l1["MLP"]=None
                teacher_layer_info_l1["feature_matrix_in"]=in_h1
                teacher_layer_info_l1["feature_matrix_out"]=out_h1
                teacher_layer_info.append(teacher_layer_info_l1)


                teacher_layer_info_l2={}
                teacher_layer_info_l2['transformation_type']=trans_order[1]
                if trans_order[0]=="MLP":
                    teacher_layer_info_l2["MLP"]=layer.fc_neigh.weight
                else:
                    teacher_layer_info_l2["MLP"]=None
                teacher_layer_info_l2["feature_matrix_in"]=in_h2
                teacher_layer_info_l2["feature_matrix_out"]=out_h2
                teacher_layer_info.append(teacher_layer_info_l2)


            if l != self.num_layers - 1:
                emb_list.append(hidden_emb)
        if save_graph_aggregation:
            self.features_before_aggr=features_before_aggr
            self.features_after_aggr=features_after_aggr
        if save_teacher_layer_info:
            self.teacher_layer_info=teacher_layer_info
        return y, emb_list
    
    def get_MLP_layers(self):
        # get the mlp part from gnn layers (sage layers with 'gcn' aggregation)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            mlp_layers.append(layer.fc_neigh.weight)
            mlp_layers.append(layer.bias)


        return mlp_layers

    def get_features_before_aggr(self):
        # [num_layers, node_num, num_features]
        return self.features_before_aggr
    
    def get_features_after_aggr(self):
        # [num_layers, node_num, num_features]
        return self.features_after_aggr
    
    def get_teacher_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.teacher_layer_info

    

class GCN(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats,save_teacher_layer_info=False):
        
        if save_teacher_layer_info:
            teacher_layer_info=[]

        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            if save_teacher_layer_info:
                h,in_hs,out_hs,trans_order = layer(g, h, save_teacher_layer_info=save_teacher_layer_info)
            else:
                h = layer(g, h)

            if save_teacher_layer_info:
                for i in range(2):
                    teacher_layer_info_layer={}
                    teacher_layer_info_layer["transformation_type"]= trans_order[i]
                    if trans_order[i]=="MLP":
                        teacher_layer_info_layer["MLP"]=layer.weight
                    else:
                        teacher_layer_info_layer["MLP"]=None
                    teacher_layer_info_layer["feature_matrix_in"]=in_hs[i]
                    teacher_layer_info_layer["feature_matrix_out"]=out_hs[i]
                    teacher_layer_info.append(teacher_layer_info_layer)


            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
        if save_teacher_layer_info:
            self.teacher_layer_info=teacher_layer_info
        return h_list, h
    
    
    def get_MLP_layers(self):
        # get the mlp part from gnn layers (sage layers with 'gcn' aggregation)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            mlp_layers.append(layer.weight)
            mlp_layers.append(layer.bias)


        return mlp_layers

    def get_teacher_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP", else will get a NoneType object,
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.teacher_layer_info

class GAT(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            num_heads=8,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=False,
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, feats,save_teacher_layer_info=False):
        
        if save_teacher_layer_info:
            teacher_layer_info=[]

            
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            
            if save_teacher_layer_info:
                h,in_hs,out_hs,trans_order = layer(g, h, save_teacher_layer_info=save_teacher_layer_info)
            else:
                h = layer(g, h)


            if save_teacher_layer_info:
                for i in range(2):
                    teacher_layer_info_layer={}
                    teacher_layer_info_layer["transformation_type"]= trans_order[i]
                    if trans_order[i]=="MLP":
                        teacher_layer_info_layer["MLP"]=layer.fc.weight
                    else:
                        teacher_layer_info_layer["MLP"]=None
                    teacher_layer_info_layer["feature_matrix_in"]=in_hs[i]
                    teacher_layer_info_layer["feature_matrix_out"]=out_hs[i]
                    teacher_layer_info.append(teacher_layer_info_layer)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
                
        if save_teacher_layer_info:
            self.teacher_layer_info=teacher_layer_info

        return h_list, h


    def get_MLP_layers(self):
        # get the mlp part from gnn layers (sage layers with 'gcn' aggregation)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            if hasattr(layer, "fc"):
                #nn.init.xavier_normal_(self.fc.weight, gain=gain)
                layer_obj=layer.fc
            else:
                ##nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
                #nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
                layer_obj=layer.fc_src
            mlp_layers.append(layer_obj.weight)
            mlp_layers.append(layer_obj.bias)


        return mlp_layers

    def get_teacher_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.teacher_layer_info

class APPNP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
            edge_drop=0.5,
            alpha=0.1,
            k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats,save_teacher_layer_info=False):
        
        if save_teacher_layer_info:
            teacher_layer_info=[]
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            if save_teacher_layer_info:
                in_h=h
                h = layer(h)
                out_h=h
            else:
                h = layer(h)

            if save_teacher_layer_info:
                teacher_layer_info_layer={}
                teacher_layer_info_layer["transformation_type"]= "MLP"
                teacher_layer_info_layer["MLP"]=layer.weight
                teacher_layer_info_layer["feature_matrix_in"]=in_h
                teacher_layer_info_layer["feature_matrix_out"]=out_h
                teacher_layer_info.append(teacher_layer_info_layer)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

                
        # GA layer
        if save_teacher_layer_info:
            in_h=h
            h = self.propagate(g, h)
            out_h=h
        else:
            h = self.propagate(g, h)

        
        if save_teacher_layer_info:
            teacher_layer_info_layer={}
            teacher_layer_info_layer["transformation_type"]= "GA"
            teacher_layer_info_layer["MLP"]=layer.weight
            teacher_layer_info_layer["feature_matrix_in"]=in_h
            teacher_layer_info_layer["feature_matrix_out"]=out_h
            teacher_layer_info.append(teacher_layer_info_layer)

            self.teacher_layer_info=teacher_layer_info

        return h_list, h

    def get_MLP_layers(self):
        # get the mlp part from gnn layers (appnp)
        # the format is  list of weight and bias: [m1weight,m1bias,m2weight,m2bias,....]
        # 
        mlp_layers=nn.ParameterList() 
        for l, layer in enumerate(self.layers):
            layer_obj=layer
            mlp_layers.append(layer_obj.weight)
            mlp_layers.append(layer_obj.bias)


        return mlp_layers

    def get_teacher_layer_info(self):
        # save 
        """data=[]
            List of dict, each dict is a transformation, the keys are:
            "transformation_type": str "GA" or "MLP",
            "MLP": MLP weight and bias tensors if transformation_type is "MLP",
            "feature_matrix_in": input feature matrix,
            "feature_matrix_out": output feature matrix,

            The transformation is the same as the order in the model
        """
        
        return self.teacher_layer_info




class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf, args, position_feature_dim=0, graph=None, byte_idx_train=None, labels_one_hot=None,from_learned_MLP_params={}):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        #print(f"model name: {conf['model_name']}")
        
        if "MLP" == conf["model_name"]:
            # origin
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                position_dim=position_feature_dim,
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
                graph=graph,
                byte_idx_train=byte_idx_train,
                labels_one_hot=labels_one_hot,
            ).to(conf["device"])
        elif "MLP_from_sequence_of_layers" == conf["model_name"] or "MLP_from_sequence_of_layers_pmp"==conf["model_name"]:
            shape_sequence_of_teacher_layers=from_learned_MLP_params['shape_sequence_of_teacher_layers']
            # transpose all if not gcn
            
            shape_sequence_of_teacher_layers=[(x[1],x[0]) for x in shape_sequence_of_teacher_layers]
            
            
            if    "same_as_teacher" in args.from_MLP_mode:
                # infer from teacher
                shape_learned_graph_aggregation_layers=[]
                for i in range(len(shape_sequence_of_teacher_layers)):
                    if shape_sequence_of_teacher_layers[i][0]>=shape_sequence_of_teacher_layers[i][1]:
                        shape_learned_graph_aggregation_layers.append((shape_sequence_of_teacher_layers[i][1],shape_sequence_of_teacher_layers[i][1]))
                    else:
                        shape_learned_graph_aggregation_layers.append((shape_sequence_of_teacher_layers[i][0],shape_sequence_of_teacher_layers[i][0]))
                    print(f"Layer {i} MLP shape {shape_sequence_of_teacher_layers[i]}, graph aggregation shape {shape_learned_graph_aggregation_layers[i]}")
            else:
                shape_learned_graph_aggregation_layers=[]
            

            if "MLP_from_sequence_of_layers" == conf["model_name"]:
                model_instance=MLP_from_sequence_of_layers


            self.encoder = model_instance(
                num_layers=args.num_layers,
                shape_sequence_of_teacher_layers=shape_sequence_of_teacher_layers,
                shape_learned_graph_aggregation_layers=shape_learned_graph_aggregation_layers,
                input_dim=args.feat_dim,
                position_dim=position_feature_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.label_dim,
                dropout_ratio=args.dropout_ratio,
                norm_type=args.norm_type,
                graph=graph,
                byte_idx_train=byte_idx_train,
                labels_one_hot=labels_one_hot,
                from_MLP_mode=args.from_MLP_mode,
                args=args,
            ).to(conf["device"])
            self.encoder.load_learned_params(from_learned_MLP_params)
            self.encoder.initialize(args.GA_init_type)
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GCN" == conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                k=conf["K"],
            ).to(conf["device"])
            


    def forward(self, data, feats,save_teacher_layer_info=False):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        
        if "MLP" in self.model_name:
            return self.encoder(feats)
        elif "GCN" in self.model_name or "GAT" in self.model_name or "APPNP" in self.model_name:
            
            return self.encoder(data, feats,save_teacher_layer_info=save_teacher_layer_info)
        else:
            return self.encoder(data, feats)[1]

    def forward_fitnet(self, data, feats):
        """
        Return a tuple (h_list, h)
        h_list: intermediate hidden representation
        h: final output
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    def inference(self, data, feats,save_graph_aggregation=False,save_teacher_layer_info=False,save_student_layer_info=False,idx_b=None,total_num=None,restart=False):
        if "SAGE" in self.model_name:
            return self.encoder.inference(data, feats,save_graph_aggregation=save_graph_aggregation,save_teacher_layer_info=save_teacher_layer_info)
        elif "MLP" in self.model_name:
            return self.encoder(feats,save_student_layer_info= save_student_layer_info,idx_b=idx_b,total_num=total_num,restart=restart)
        else:
            return self.forward(data, feats,save_teacher_layer_info=save_teacher_layer_info)

    def encode_teacher4kd(self, teacher_feat):
        return self.encoder.encode_teacher4kd(teacher_feat)

    def encode_mlp4kd(self, mlp_feat):
        return self.encoder.encode_mlp4kd(mlp_feat)


    def encode_mlp4attn_distill(self, mlp_feat,layer_num):
        return self.encoder.encode_mlp4attn_distill(mlp_feat,layer_num)
    
    def encode_mlp4lp(self, mlp_feat,layer_num):
        return self.encoder.encode_mlp4lp(mlp_feat,layer_num)

    def encode_mlp4etype_pred(self, mlp_feat,layer_num):
        return self.encoder.encode_mlp4etype_pred(mlp_feat,layer_num)
