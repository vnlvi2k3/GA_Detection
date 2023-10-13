import torch
from layers import Encoder, Inlier_Decoder, Outlier_Decoder

class gcnn(nn.Module):
    def __init__(self, args):
        super(gcnn, self).__init__()
        self.embed_dim = args.embed_dim
        self.dim1 = args.decoder_dim1
        self.dim2 = args.decoder_dim2
        self.output_dim = args.output_dim
        
        self.encoder = Encoder(args)
        self.Inlier_decoder = Inlier_Decoder(input_dim = self.embed_dim * 3, dim1 = self.dim1, 
                                             dim2 = self.dim2, output_dim = self.output_dim)
        self.Outlier_decoder = Outlier_Decoder(input_dim = self.embed_dim * 3, dim1 = self.dim1, 
                                             dim2 = self.dim2, output_dim = self.output_dim)
        self.normal_loss = torch.nn.MSELoss(reduction='mean')
        self.recon_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, X, features, labels):
        normal_idx = torch.where(labels == 0)[0].tolist()
        unlabeled_idx = torch.where(labels != 0)[0].tolist()
        src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats = X
        
        embs = self.encoder(features, src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats)
        recon_normal_feats = self.inlier_decoder(embs[normal_idx])
        recon_unlabeled_feats_in = self.inlier_decoder(embs[unlabeled_idx])
        recon_unlabeled_feats_out = self.outlier_decoder(embs[unlabeled_idx])
        
        normal_loss = self.cal_normal_loss(recon_normal_feats, features[normal_idx,:])
        pred, in_loss, out_loss = self.cal_recon_loss(recon_unlabeled_feats_in, recon_unlabeled_feats_out, features[unlabeled_idx,:])
        return pred, normal_loss, in_loss, out_loss
        
    def cal_normal_loss(self, recon_normal_feats, normal_feats):
        return self.normal_loss(recon_normal_feats, normal_feats)
    
    def cal_recon_loss(self, recon_feats_in, recon_feats_out, unlabeled_feats):
        in_recon_loss = self.recon_loss(recon_feats_in, unlabeled_feats).mean(axis=1)
        out_recon_loss = self.recon_loss(recon_feats_out, unlabeled_feats).mean(axis=1)
        
        pred_labels = torch.where(in_recon_loss < out_recon_loss, 0., 1.)
        in_loss = torch.mul((1. - pred_labels), in_recon_loss).mean()
        out_loss = torch.mul(pred_labels, out_recon_loss).mean()
        
        return pred_labels, in_loss, out_loss