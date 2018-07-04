import numpy as np
from torch.utils.data.dataset import Dataset


class SparseDataset(Dataset):
    """
    A dataset subclass for sparse data.

    data_tensor: scipy sparse matrix of size [N x D]
    tfidf:
    """

    def __init__(self, data_tensor, tfidf):
        self.data_tensor = data_tensor
        self.N = self.data_tensor.shape[0]
        self.tfidf = tfidf

    def __getitem__(self, index):
        x = self.data_tensor[index].toarray()
        if self.tfidf is None:
            return x.astype('float32'), np.zeros(1)
        idf = x * self.tfidf
        return x.astype('float32'), (idf/np.sqrt((idf**2).sum(1,keepdims=True))).astype('float32')

    def __len__(self):
        return self.N
