import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def data_loader(data_path):
    data = loadmat(data_path)
    data = data['action_features']
    features=torch.Tensor(data)
    labels = data[:,-1] 
    index = list(range(len(labels)))
    # spliting train and test data
    idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.60, random_state = 2, shuffle = True)

    train = {
        'idx': idx_train,
        'feats': features,
        'labels': labels
    }
    test = {
        'idx': idx_test,
        'feats': features[idx_test],
        'labels': labels[idx_test]
    }
    return train, test