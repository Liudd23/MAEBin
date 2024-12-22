
from collections import namedtuple, Counter
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    # 保存节点特征和边特征
    feat = graph.ndata["feat"]

    # 将单向图转化为双向图
    graph = dgl.to_bidirected(graph)

  
    # 恢复节点特征
    graph.ndata["feat"] = feat

    # 移除自环并添加自环
    graph = graph.remove_self_loop().add_self_loop()

    # # 为自环添加边特征（通常可以设置为1或其他默认值）
    # num_self_loops = graph.num_nodes()
    # self_loop_weights = torch.ones(num_self_loops, dtype=weight.dtype, device=weight.device)
    # graph.edata['_edge_weight'] = torch.cat([graph.edata['_edge_weight'], self_loop_weights], dim=0)

    # 优化图的存储格式
    graph.create_formats_()

    return graph

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)

def create_graph_from_txt(file_path):
    src_list = []
    dst_list = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # 跳过格式不正确的行
            src_node = int(parts[0])
            dst_node = int(parts[1])
            src_list.append(src_node)
            dst_list.append(dst_node)

    src = torch.tensor(src_list)
    dst = torch.tensor(dst_list)

    graph = dgl.graph((src, dst))
    return graph

def tsv_to_dgl_features(tsv_path):
    print("开始进行tsv文件转化")
    """
    将 TSV 文件中的数据转化为 DGL 图的节点特征。
    - 去掉第一行（标题）和第一列（索引）。
    
    参数:
        tsv_path (str): TSV 文件的路径。
    
    返回:
        g (DGLGraph): 包含节点特征的 DGL 图。
    """
    data = pd.read_csv(tsv_path, sep='\t', header=0, index_col=0)
    
    # 转换为 PyTorch Tensor
    features = torch.tensor(data.values, dtype=torch.float32)
    return features

def load_custom_dataset(graph,feature):
    print("开始加载客制化数据集！！！")
    # 创建图
    graph = create_graph_from_txt(graph)
    print("创建图成功")
    
    # 读取特征
    feature = tsv_to_dgl_features(feature)
    print("特征读取成功！！")
    print(f"特征形状: {feature.shape}")

    # 检查特征和节点数量
    num_nodes = graph.num_nodes()
    print("--------------------------")
    print(num_nodes)
    num_features = feature.shape[0]  # 特征数量（行数）
    feature_dim = feature.shape[1]
    if num_features > num_nodes:
        # 找出需要添加的节点数量
        extra_nodes = num_features - num_nodes

        # 添加新节点
        graph.add_nodes(extra_nodes)

        # 为新节点添加自环
        new_node_ids = torch.arange(num_nodes, num_nodes + extra_nodes)
        graph.add_edges(new_node_ids, new_node_ids)

        print(f"添加了 {extra_nodes} 个节点，并为其加上了自环。")
    elif num_features < num_nodes:
        # 如果特征数量少于节点数，用零特征填充
        padding = torch.zeros((num_nodes - num_features, feature.shape[1]))
        feature = torch.cat([feature, padding], dim=0)
        print(f"特征不足，已用零特征填充到 {num_nodes} 行。")

    # 标准化特征
    feature = scale_feats(feature)

    # 分配节点特征
    graph.ndata["feat"] = feature

    # 检查结果
    print(f"最终图节点数: {graph.num_nodes()}, 特征形状: {graph.ndata['feat'].shape}")

    # 可选的预处理步骤
    graph = preprocess(graph)
    return graph, feature_dim




def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
