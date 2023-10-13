from collections import defaultdict
import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn

data = loadmat('preprocessed_data/example.mat')
relation = data['bipartite_action'].astype(int)
edge_features = data['action_features'] 
item_features = data['item_features_matrix'] 
user_features = data['user_features_matrix'] 
edge_user_features = data['edge_user_features_matrix'] 
edge_item_features = data['edge_item_features_matrix'] 

def get_dict(relation):
    user_dict = defaultdict(list)
    item_dict = defaultdict(list)
    for i in range(relation.shape[0]):
        user_dict[relation[i, 1]].append(relation[i, 0])
        item_dict[relation[i, 2]].append(relation[i, 0])
    return user_dict, item_dict

def get_neig(src_idx, dictionary, hot, edge_node_feats):
    nei_idx = []
    for idx in src_idx:
        replace = False
        if len(dictionary[idx]) < hot:
            replace = True
        chosen_nodes = np.random.choice(dictionary[idx], hot, replace = replace).tolist()
        nei_idx.extend(chosen_nodes)
    nei_feats = edge_node_feats[nei_idx]
    return nei_idx, nei_feats
    
def get_neighbors(edge_idx, batch_size, user_dict, item_dict, hot1, hot2):
    src_idx = relation[edge_idx, 1].tolist()
    dst_idx = relation[edge_idx, 2].tolist()
    
    src_feats = user_features[src_idx]
    dst_feats = item_features[dst_idx]
    
    # Sampling the 1-hop neighbors for src nodes, src nodes is user -> so hop1 will be item
    src_hot1_idx, src_hot1_feats = get_neig(src_idx, user_dict, hot1, edge_item_features)
    src_hot1_feats = src_hot1_feats.reshape(batch_size, hot1, -1)
    src_hot1_node_idx = relation[src_hot1_idx, 2].tolist()
    
    #Sampling the 2-hop neighbors for scr nodes, hot1 is item -> hot2 is user
    _, src_hot2_feats = get_neig(src_hot1_node_idx, item_dict, hot2, edge_user_features)
    src_hot2_feats = src_hot2_feats.reshape(batch_size, hot1, hot2, -1)
    
    #Sampling the 1-hop neighbors for dst nodes, dst is item -> so hot1 is user
    dst_hot1_idx, dst_hot1_feats = get_neig(dst_idx, item_dict, hot1, edge_user_features)
    dst_hot1_feats = dst_hot1_feats.reshape(batch_size, hot1, -1)
    dst_hot1_node_idx = relation[dst_hot1_node_idx, 1].tolist()
    
    #Sampling the 2-hop neighbors for dst nodes -> item
    _, dst_hot2_feats = get_neig(dst_hot1_node_idx, user_dict, hot2, edge_item_features)
    dst_hot2_feats = dst_hot2_feats.reshape(batch_size, hot1, hot2, -1)
    
    src_feats = torch.FloatTensor(src_feats)
    dst_feats = torch.FloatTensor(dst_feats)
    
    src_hot1_feats = torch.FloatTensor(src_hot1_feats)
    dst_hot1_feats = torch.FloatTensor(dst_hot1_feats)
    
    src_hot2_feats = torch.FloatTensor(src_hot2_feats)
    dst_hot2_feats = torch.FloatTensor(dst_hot2_feats)
    
    return src_feats, dst_feats, src_hot1_feats, src_hot2_feats, dst_hot1_feats, dst_hot2_feats

def initialize_model(model, device, load_save_file=False, gpu=True):
    if load_save_file:
        if not gpu:
            model.load_state_dict(
                torch.load(load_save_file, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(load_save_file))
    # else:
    #     for param in model.parameters():
    #         if param.dim() == 1:
    #             continue
    #             nn.init.constant(param, 0)
    #         else:
    #             # nn.init.normal(param, 0.0, 0.15)
    #             nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    return model