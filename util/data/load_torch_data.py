import numpy
import scipy
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from sparse_dataset import SparseDataset


def load_torch_data(load_data_func):
    """Wrapper around load_data to instead use pytorch data loaders."""

    def torch_loader(dataset, data_path, batch_size, shuffle=True, cuda_device=None, num_workers=1):
        (train_data, val_data), (train_labels, val_labels), label_names = load_data_func(dataset, data_path)

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda_device is not None else {}
        kwargs['drop_last'] = True

        if type(train_data) == numpy.ndarray:
            train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
            val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_labels))
        elif type(train_data) == scipy.sparse.csr.csr_matrix:
            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf_trans = TfidfTransformer(norm=None)
            tfidf_trans.fit(train_data)
            train_dataset = SparseDataset(train_data, tfidf_trans.idf_)
            val_dataset = SparseDataset(val_data, tfidf_trans.idf_)
        else:
            train_dataset = torchvision.datasets.ImageFolder(train_data)
            val_dataset = torchvision.datasets.ImageFolder(val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, label_names

    return torch_loader
