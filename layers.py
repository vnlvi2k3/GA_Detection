import torch 
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F 

class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
    def forward(self, features):
        return torch.matmul(features, self.weight)

class SageGCN(nn.Module):
    def __init__(self, src_dim, nei_dim, output_dim, dropout=0.5):
        super(SageGCN, self).__init__()
        self.src_dim = src_dim
        self.nei_dim = nei_dim 
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator = MeanAggregator(self.nei_dim, self.output_dim)
        self.weight = nn.Prameter(torch.FloatTensor(self.src_dim, self.output_dim))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
    def forward(self, src_features, nei_features):
        embed_nei = self.aggregator(nei_features)
        embed_src = torch.matmul(src_features, self.weight)
        
        embed_feats = torch.cat([embed_src, embed_nei], dim=-1)
        embed_feats = F.dropout(embed_feats, self.dropout, training=self.training)#self.training
        return F.relu(embed_feats)

        
#Two convolution layers
#[src] <-- [dst] <-- [src] <-- [[edge]] --> [dst] --> [src] --> [dst]
class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.src_dim = args.src_dim 
        self.dst_dim = args.dst_dim
        self.edge_dim = args.edge_dim
        self.src_hidden_dim1 = args.src_hidden_dim1
        self.dst_hidden_dim1 = args.dst_hidden_dim1
        self.embed_dim = args.embed_dim
        
        self.src_agg1 = SageGCN(self.src_dim, self.dst_dim + self.edge_dim, self.src_hidden_dim1)
        self.src_agg2 = SageGCN(self.src_hidden_dim1, self.dst_hidden_dim1, self.embed_dim)
        self.dst_agg1 = SageGCN(self.dst_dim, self.src_dim + self.edge_dim, self.dst_hidden_dim1)
        self.dst_agg2 = SageGCN(self.dst_hidden_dim1, self.dst_hidden_dim1, self.embed_dim)
    
    def forward(self, src_feats, src_hop1_feats, src_hop2_feats, dst_feats, dst_hop1_feats, dst_hop2_feats):
        src_hop1_mean = src_hop1_feats.mean(dim=1)
        dst_hop1_mean = dst_hop1_feats.mean(dim=1)

        #src_hop1 is destination, dst_hop2 is source
        src_feats_agg1 = self.src_agg1(src_feats, src_hop1_mean)
        dst_feats_agg1 = self.dst_agg1(dst_feats, dst_hop1_mean)
        
        #src_hop2 is source, dst_hop2 is destination
        src_hop2_mean = src_hop2_feats.mean(dim=2)
        dst_hop2_mean = dst_hop2_feats.mean(dim=2)
        
        src_hop1_agg1 = self.dst_agg1(src_hop1_feats[:,:,:self.dst_dim], src_hop2_mean)
        dst_hop1_agg1 = self.src_agg1(dst_hop1_feats[:,:,:self.src_dim], dst_hop2_mean)
        
        src_agg1_mean = src_hop1_agg1.mean(dim=1)
        dst_agg1_mean = dst_hop1_agg1.mean(dim=1)
        
        src_feats_agg2 = self.src_agg2(src_feats_agg1, src_agg1_mean)
        dst_feats_agg2 = self.dst_agg2(dst_feats_agg1, dst_agg1_mean)
        
        return src_feats_agg2, dst_feats_agg2
    
class Edge_Encoder(nn.Module):
    def __init__(self, input_dim, dim1, output_dim):
        super(Edge_Encoder, self).__init__()
        self.input_dim = input_dim
        self.dim1 = dim1
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, self.dim1)
        self.layer2 = nn.Linear(self.dim1, self.output_dim)
        
    def forward(self, edge_feats):
        embs = F.relu(self.layer1(edge_feats))
        return self.layer2(embs)

class Inlier_Decoder(nn.Module):
    def __init__(self, input_dim, dim1, dim2, output_dim):
        super(Inlier_Decoder, self).__init__()
        self.input_dim = input_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, self.dim1)
        self.layer2 = nn.Linear(self.dim1, self.dim2)
        self.layer3 = nn.Linear(self.dim2, self.output_dim)
    
    def forward(self, embeddings):
        feats = F.relu(self.layer1(embeddings))
        feats = F.relu(self.layer2(feats))
        return self.layer3(feats)

class Outlier_Decoder(nn.Module):
    def __init__(self, input_dim, dim1, dim2, output_dim):
        super(Outlier_Decoder, self).__init__()
        self.input_dim = input_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, self.dim1)
        self.layer2 = nn.Linear(self.dim1, self.dim2)
        self.layer3 = nn.Linear(self.dim2, self.output_dim)
    
    def forward(self, embeddings):
        feats = F.relu(self.layer1(embeddings))
        feats = F.relu(self.layer2(feats))
        return self.layer3(feats)
    
class Encoder(nn.Module):
    # concateing embeddings of edges, users, and items
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.src_dim = args.src_dim 
        self.dst_dim = args.dst_dim
        self.edge_dim = args.edge_dim
        self.src_hidden_dim1 = args.src_hidden_dim1
        self.dst_hidden_dim1 = args.dst_hidden_dim1
        self.embed_dim = args.embed_dim
        
        self.GNN_Encoder = GCN(args)
        self.Feats_Encoder = Edge_Encoder(self.edge_dim, 6, 8)
    
    def forward(self, edge_feats, src_feats, src_hop1_feats, src_hop2_feats, dst_feats, dst_hop1_feats, dst_hop2_feats):
        user_embedding, item_embedding = self.GNN_Encoder(src_feats, src_hop1_feats, src_hop2_feats, dst_feats, dst_hop1_feats, dst_hop2_feats)
        edge_embedding = self.Feats_Encoder(edge_feats)
        embs = torch.cat([user_embedding, item_embedding, edge_embedding], dim=1)
        return embs