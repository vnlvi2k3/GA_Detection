import argparse
import torch
import random
from datasets import data_loader
from utils import get_dict
from model import gcnn
from utils import get_neighbors, initialize_model
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type = int, default = 40, help = "number of epoches")
parser.add_argument("--lr", type = float, default = 0.001, help = "learning rate")
parser.add_argument("--batch_size", type = int, default = 200, help = "batch_size")
parser.add_argument("--hot1", type = int, default = 4, help = "1-hop neighbors")
parser.add_argument("--hot2", type = int, default = 4, help = "2-hop neighbors")
parser.add_argument("--src_dim", type = int, default = 27, help = "encoder source dim")
parser.add_argument("--dst_dim", type = int, default = 4, help = "encoder dst dim")
parser.add_argument("--edge_dim", type = int, default = 4, help = "encoder edge dim")
parser.add_argument("--src_hidden_dim1", type = int, default = 16, help = "1-hop source dim")
parser.add_argument("--dst_hidden_dim1", type = int, default = 4, help = "1-hop dst dim")
parser.add_argument("--embed_dim", type = int, default = 8, help = "output node embeddings")
parser.add_argument("--data_path", type=str, default="preprocessed_data/example.mat", help="data path")

parser.add_argument("--decoder_dim1", type = int, default = 16, help = "decoder hidden dim1")
parser.add_argument("--decoder_dim2", type = int, default = 8, help = "decoder hidden dim2")
parser.add_argument("--output_dim", type = int, default = 4, help = "decoder output dim")
parser.add_argument("--ckpt", help="Load ckpt file", type=str, default="")

def main(args):
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size

    train_loader, test_loader = data_loader()
    user_dict, item_dict = get_dict()
    train_idx = train_loader['idx']
    feats_data = train_loader['feats']
    num_batches = int(len(train_idx) / batch_size)

    model = gcnn(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model, device, load_save_file=args.ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        random.shuffle(train_idx)
        model.train()

        for batch in range(num_batches):
            i_start = batch * args.batch_size
            i_end = min((batch + 1) * batch_size, len(train_idx))
            batch_nodes = train_idx[i_start : i_end]
            batch_size = len(batch_nodes)

            features = feats_data[batch_nodes][:, :-1]
            labels = feats_data[batch_nodes][:, -1]
            src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats = get_neighbors(batch_nodes, batch_size, user_dict, item_dict, args.hot1, args.hot2)
            
            features, labels, src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats = (
                features.to(device),
                labels.to(device),
                src_feats.to(device),
                dst_feats.to(device),
                src_hop1_feats.to(device),
                src_hop2_feats.to(device),
                dst_hop1_feats.to(device),
                dst_hop2_feats.to(device), 
            )
            pred, normal_loss, in_loss, out_loss = model(
                X = (src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats), 
                features = features, labels = labels
            )
            loss = normal_loss + in_loss + out_loss
            print('epoch:{} batch:{} --> loss={}'.format(epoch, 0, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Testing
        model.eval()
        test_idx = test_loader['idx']
        features = test_loader['feats'][0, :4]
        labels = test_loader['labels']
        batch_size = len(labels)
        src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats = get_neighbors(batch_nodes, batch_size, user_dict, item_dict, args.hot1, args.hot2)
        features, labels, src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats = (
                features.to(device),
                labels.to(device),
                src_feats.to(device),
                dst_feats.to(device),
                src_hop1_feats.to(device),
                src_hop2_feats.to(device),
                dst_hop1_feats.to(device),
                dst_hop2_feats.to(device), 
            )
        pred, normal_loss, in_loss, out_loss = model(
            X = (src_feats, dst_feats, src_hop1_feats, src_hop2_feats, dst_hop1_feats, dst_hop2_feats), 
            features = features, labels = labels
        )
        fraud_score = (in_loss / (in_loss + out_loss)).detach().numpy()
        auc = roc_auc_score(labels, fraud_score)
        print('ROC AUC score:{:0.4f}'.format(auc))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    main(args)