import os
import urllib
import struct
import gzip
import zipfile
import tarfile
import numpy as np
from scipy.io import loadmat

from load_torch_data import load_torch_data

# todo: add label names to omniglot
# todo: add labels to static binarized MNIST, omniglot


@load_torch_data
def load_data(dataset, data_path):

    """
    Downloads and loads a variety of benchmark image datasets.
    Note that some datasets do not have labels and label names.

    # Arguments
        dataset: a string from one of the datasets supported below
        data_path: a path string to the location of the datasets

    # Returns
        (train, val): for small datasets, these are data tensors of
                      size  N x H x W x C. for larger datasets, these
                      are paths to the image directories
        (train_labels, val_labels): lists of label indices for the images.
                                    note that some datasets do not have labels.
        label_names: lists of (human readable) string names for the
                     label indices. note that some datasets do not have
                     label names.
    """

    assert os.path.exists(data_path), 'Data path not found. Please specify a valid path.'

    print 'Loading data...'
    train = val = None
    train_labels = val_labels = None
    label_names = None

    def unpickle(file_name):
        import cPickle
        fo = open(file_name, 'rb')
        result = cPickle.load(fo)
        fo.close()
        return result

    if dataset in ['binarized_MNIST', 'MNIST']:
        if not os.path.exists(os.path.join(data_path, 'MNIST')):
            os.makedirs(os.path.join(data_path, 'MNIST'))

        def load_mnist_images_np(imgs_filename):
            with open(imgs_filename, 'rb') as f:
                f.seek(4)
                nimages, rows, cols = struct.unpack('>iii', f.read(12))
                images = np.fromfile(f, dtype=np.dtype(np.ubyte)).astype('float32').reshape((nimages,rows,cols,1))
            return images

        def load_mnist_labels_np(labels_filename):
            with open(labels_filename, 'rb') as f:
                f.seek(4)
                nlabels = struct.unpack('>i', f.read(4))
                labels = np.fromfile(f, dtype=np.dtype(np.ubyte)).astype('float32').reshape((nlabels))
            return labels

        if not os.path.exists(os.path.join(data_path, 'MNIST', 'train-images-idx3-ubyte')):
            print 'Downloading MNIST training images...'
            urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', os.path.join(data_path, 'MNIST', 'train-images-idx3-ubyte.gz'))
            with gzip.open(os.path.join(data_path, 'MNIST', 'train-images-idx3-ubyte.gz'), 'rb') as f:
                file_content = f.read()
            with open(os.path.join(data_path, 'MNIST', 'train-images-idx3-ubyte'), 'wb') as f:
                f.write(file_content)
            os.remove(os.path.join(data_path, 'MNIST', 'train-images-idx3-ubyte.gz'))
        train = load_mnist_images_np(os.path.join(data_path, 'MNIST', 'train-images-idx3-ubyte'))

        if not os.path.exists(os.path.join(data_path, 'MNIST', 'train-labels-idx1-ubyte')):
            print 'Downloading MNIST training labels...'
            urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', os.path.join(data_path, 'MNIST', 'train-labels-idx1-ubyte.gz'))
            with gzip.open(os.path.join(data_path, 'MNIST', 'train-labels-idx1-ubyte.gz'), 'rb') as f:
                file_content = f.read()
            with open(os.path.join(data_path, 'MNIST', 'train-labels-idx1-ubyte'), 'wb') as f:
                f.write(file_content)
            os.remove(os.path.join(data_path, 'MNIST', 'train-labels-idx1-ubyte.gz'))
        train_labels = load_mnist_labels_np(os.path.join(data_path, 'MNIST', 'train-labels-idx1-ubyte'))

        if not os.path.exists(os.path.join(data_path, 'MNIST', 't10k-images-idx3-ubyte')):
            print 'Downloading MNIST validation data...'
            urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', os.path.join(data_path, 'MNIST', 't10k-images-idx3-ubyte.gz'))
            with gzip.open(os.path.join(data_path, 'MNIST', 't10k-images-idx3-ubyte.gz'), 'rb') as f:
                file_content = f.read()
            with open(os.path.join(data_path, 'MNIST', 't10k-images-idx3-ubyte'), 'wb') as f:
                f.write(file_content)
            os.remove(os.path.join(data_path, 'MNIST', 't10k-images-idx3-ubyte.gz'))
        val = load_mnist_images_np(os.path.join(data_path, 'MNIST', 't10k-images-idx3-ubyte'))

        if not os.path.exists(os.path.join(data_path, 'MNIST', 't10k-labels-idx1-ubyte')):
            print 'Downloading MNIST validation labels...'
            urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', os.path.join(data_path, 'MNIST', 't10k-labels-idx1-ubyte.gz'))
            with gzip.open(os.path.join(data_path, 'MNIST', 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
                file_content = f.read()
            with open(os.path.join(data_path, 'MNIST', 't10k-labels-idx1-ubyte'), 'wb') as f:
                f.write(file_content)
            os.remove(os.path.join(data_path, 'MNIST', 't10k-labels-idx1-ubyte.gz'))
        val_labels = load_mnist_labels_np(os.path.join(data_path, 'MNIST', 't10k-labels-idx1-ubyte'))

        label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    elif dataset == 'static_binarized_MNIST':
        if not os.path.exists(os.path.join(data_path, 'static_binarized_MNIST')):
            os.makedirs(os.path.join(data_path, 'static_binarized_MNIST'))

        if not os.path.exists(os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_train.amat')):
            print 'Downloading binarized MNIST training data...'
            urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat', os.path.join(data_path,'static_binarized_MNIST', 'binarized_mnist_train.amat'))

        with open(os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_train.amat')) as f:
            lines = f.readlines()
        _train1 = np.array([[int(i) for i in line.split()] for line in lines]).astype('float32')

        if not os.path.exists(os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_valid.amat')):
            print 'Downloading binarized MNIST validation data...'
            urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat', os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_valid.amat'))

        with open(os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_valid.amat')) as f:
            lines = f.readlines()
        _train2 = np.array([[int(i) for i in line.split()] for line in lines]).astype('float32')

        train = np.concatenate([_train1, _train2], axis=0).reshape((-1, 28, 28))

        if not os.path.exists(os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_test.amat')):
            print 'Downloading binarized MNIST testing data...'
            urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat', os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_test.amat'))

        with open(os.path.join(data_path, 'static_binarized_MNIST', 'binarized_mnist_test.amat')) as f:
            lines = f.readlines()
        val = np.array([[int(i) for i in line.split()] for line in lines]).astype('float32').reshape((-1, 28, 28))

        label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    elif dataset == 'omniglot':
        if not os.path.exists(os.path.join(data_path, 'omniglot')):
            os.makedirs(os.path.join(data_path, 'omniglot'))

        if not os.path.exists(os.path.join(data_path, 'omniglot', 'chardata.mat')):
            print 'Downloading Omniglot images_background.zip...'
            urllib.urlretrieve('https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat', os.path.join(data_path, 'omniglot', 'chardata.mat'))
        data = loadmat(os.path.join(data_path, 'omniglot', 'chardata.mat'))
        train = data['data'].swapaxes(0,1).reshape((-1, 28, 28))
        val = data['testdata'].swapaxes(0,1).reshape((-1, 28, 28))

    elif dataset == 'caltech_101_silhouettes':
        if not os.path.exists(os.path.join(data_path, 'caltech_101_silhouettes')):
            os.makedirs(os.path.join(data_path, 'caltech_101_silhouettes'))
        if not os.path.exists(os.path.join(data_path, 'caltech_101_silhouettes', 'caltech101_silhouettes_28_split1.mat')):
            print 'Downloading Caltech 101 Silhouettes...'
            urllib.urlretrieve('https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat', os.path.join(data_path, 'caltech_101_silhouettes', 'caltech101_silhouettes_28_split1.mat'))
        data = loadmat(os.path.join(data_path, 'caltech_101_silhouettes', 'caltech101_silhouettes_28_split1.mat'))
        train = np.concatenate([data['train_data'], data['val_data']], axis=0).astype('float32').reshape((-1, 28, 28))
        train_labels = np.concatenate([data['train_labels'], data['val_labels']], axis=0).astype('float32').reshape(-1)
        val = data['test_data'].astype('float32').reshape((-1, 28, 28))
        val_labels = data['test_labels'].astype('float32').reshape(-1)
        label_names = data['classnames'].reshape(-1)
        label_names = [str(label_names[i][0]) for i in range(label_names.shape[0])]

    elif dataset == 'CIFAR_10':
        if not os.path.exists(os.path.join(data_path, 'CIFAR_10')):
            os.makedirs(os.path.join(data_path, 'CIFAR_10'))
        if not os.path.exists(os.path.join(data_path, 'CIFAR_10', 'cifar-10-batches-py')):
            print 'Downloading CIFAR_10...'
            urllib.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', os.path.join(data_path, 'CIFAR_10', 'cifar-10-python.tar.gz'))
            print 'Extracting CIFAR_10 tar file...'
            tar = tarfile.open(os.path.join(data_path, 'CIFAR_10', 'cifar-10-python.tar.gz'))
            tar.extractall(os.path.join(data_path, 'CIFAR_10'))
            tar.close()
        _train = [unpickle(os.path.join(data_path, 'CIFAR_10', 'cifar-10-batches-py', 'data_batch_' + str(i + 1))) for i in range(5)]
        train = np.concatenate([_train[i]['data'] for i in range(5)]).astype('float32').reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
        train_labels = np.concatenate([_train[i]['labels'] for i in range(5)])
        _val = unpickle(os.path.join(data_path, 'CIFAR_10', 'cifar-10-batches-py', 'test_batch'))
        val = _val['data'].astype('float32').reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
        val_labels = np.array(_val['labels'])
        label_dict = unpickle(os.path.join(data_path, 'CIFAR_10', 'cifar-10-batches-py', 'batches.meta'))
        label_names = label_dict['label_names']

    elif dataset == 'CIFAR_100':
        if not os.path.exists(os.path.join(data_path, 'CIFAR_100')):
            os.makedirs(os.path.join(data_path, 'CIFAR_100'))
        if not os.path.exists(os.path.join(data_path, 'CIFAR_100', 'cifar-100-python')):
            print 'Downloading CIFAR_100...'
            urllib.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', os.path.join(data_path, 'CIFAR_100', 'cifar-100-python.tar.gz'))
            print 'Extracting CIFAR_100 tar file...'
            tar = tarfile.open(os.path.join(data_path, 'CIFAR_100', 'cifar-100-python.tar.gz'))
            tar.extractall(os.path.join(data_path, 'CIFAR_100'))
            tar.close()
        _train = unpickle(os.path.join(data_path, 'CIFAR_100', 'cifar-100-python', 'train'))
        train = _train['data'].astype('float32').reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
        train_labels = np.array(_train['fine_labels'])
        _val = unpickle(os.path.join(data_path, 'CIFAR_100', 'cifar-100-python', 'test'))
        val = _val['data'].astype('float32').reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
        val_labels = np.array(_val['fine_labels'])
        label_dict = unpickle(os.path.join(data_path, 'CIFAR_100', 'cifar-100-python', 'meta'))
        label_names = label_dict['fine_label_names']

    elif dataset == 'SVHN':
        if not os.path.exists(os.path.join(data_path, 'SVHN')):
            os.makedirs(os.path.join(data_path, 'SVHN'))
        if not os.path.exists(os.path.join(data_path, 'SVHN','train_32x32.mat')):
            print 'Downloading SVHN train...'
            urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', os.path.join(data_path, 'SVHN', 'train_32x32.mat'))
        data_labels = loadmat(os.path.join(data_path, 'SVHN', 'train_32x32.mat'))
        train = data_labels['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1).astype('float32')
        train_labels = data_labels['y'].reshape(-1).astype('float32')
        if not os.path.exists(os.path.join(data_path, 'SVHN', 'test_32x32.mat')):
            print 'Downloading SVHN test...'
            urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', os.path.join(data_path, 'SVHN', 'test_32x32.mat'))
        data_labels = loadmat(os.path.join(data_path, 'SVHN', 'test_32x32.mat'))
        val = data_labels['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1).astype('float32')
        val_labels = data_labels['y'].reshape(-1).astype('float32')

        label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    elif dataset == 'imagenet_32':
        if not os.path.exists(os.path.join(data_path, 'imagenet_32')):
            os.makedirs(os.path.join(data_path, 'imagenet_32'))
        if not os.path.exists(os.path.join(data_path, 'imagenet_32', 'train_32x32')):
            print 'Downloading ImageNet 32 x 32 train data...'
            urllib.urlretrieve('http://image-net.org/small/train_32x32.tar', os.path.join(data_path, 'imagenet_32', 'train_32x32.tar'))
            tar = tarfile.open(os.path.join(data_path, 'imagenet_32', 'train_32x32.tar'))
            print 'Extracting ImageNet 32 x 32 tar file...'
            tar.extractall(os.path.join(data_path, 'imagenet_32'))
            tar.close()
            os.remove(os.path.join(data_path, 'imagenet_32', 'train_32x32.tar'))
            print 'Moving images into directory...'
            os.makedirs(os.path.join(data_path, 'imagenet_32', 'train_32x32', 'images'))
            for _, _, files in os.walk(os.path.join(data_path, 'imagenet_32', 'train_32x32')):
                root = os.path.join(data_path, 'imagenet_32', 'train_32x32')
                for f in files:
                    if os.path.exists(os.path.join(root, f)):
                        os.rename(os.path.join(root, f), os.path.join(root, 'images', f))
        train = os.path.join(data_path, 'imagenet_32', 'train_32x32')

        if not os.path.exists(os.path.join(data_path, 'imagenet_32', 'valid_32x32')):
            print 'Downloading ImageNet 32 x 32 validation data...'
            urllib.urlretrieve('http://image-net.org/small/valid_32x32.tar', os.path.join(data_path, 'imagenet_32', 'valid_32x32.tar'))
            tar = tarfile.open(os.path.join(data_path, 'imagenet_32', 'valid_32x32.tar'))
            print 'Extracting ImageNet 32 x 32 tar file...'
            tar.extractall(os.path.join(data_path, 'imagenet_32'))
            tar.close()
            os.remove(os.path.join(data_path, 'imagenet_32', 'valid_32x32.tar'))
            print 'Moving images into directory...'
            os.makedirs(os.path.join(data_path, 'imagenet_32', 'valid_32x32', 'images'))
            for _, _, files in os.walk(os.path.join(data_path, 'imagenet_32', 'valid_32x32')):
                root = os.path.join(data_path, 'imagenet_32', 'valid_32x32')
                for f in files:
                    if os.path.exists(os.path.join(root, f)):
                        os.rename(os.path.join(root, f), os.path.join(root, 'images', f))
        val = os.path.join(data_path, 'imagenet_32', 'valid_32x32')

    elif dataset == 'imagenet_64':
        if not os.path.exists(os.path.join(data_path, 'imagenet_64')):
            os.makedirs(os.path.join(data_path, 'imagenet_64'))
        if not os.path.exists(os.path.join(data_path, 'imagenet_64', 'train_64x64')):
            print 'Downloading ImageNet 64 x 64 train data...'
            urllib.urlretrieve('http://image-net.org/small/train_64x64.tar', os.path.join(data_path, 'imagenet_64', 'train_64x64.tar'))
            tar = tarfile.open(os.path.join(data_path, 'imagenet_64', 'train_64x64.tar'))
            print 'Extracting ImageNet 64 x 64 tar file...'
            tar.extractall(os.path.join(data_path, 'imagenet_64'))
            tar.close()
            os.remove(os.path.join(data_path, 'imagenet_64', 'train_64x64.tar'))
            print 'Moving images into directory...'
            os.makedirs(os.path.join(data_path, 'imagenet_64', 'train_64x64', 'images'))
            for _, _, files in os.walk(os.path.join(data_path, 'imagenet_64', 'train_64x64')):
                root = os.path.join(data_path, 'imagenet_64', 'train_64x64')
                for f in files:
                    if os.path.exists(os.path.join(root, f)):
                        os.rename(os.path.join(root, f), os.path.join(root, 'images', f))
        train = os.path.join(data_path, 'imagenet_64', 'train_64x64')

        if not os.path.exists(os.path.join(data_path, 'imagenet_64', 'valid_64x64')):
            print 'Downloading ImageNet 64 x 64 validation data...'
            urllib.urlretrieve('http://image-net.org/small/valid_64x64.tar', os.path.join(data_path, 'imagenet_64', 'valid_64x64.tar'))
            tar = tarfile.open(os.path.join(data_path, 'imagenet_64', 'valid_64x64.tar'))
            print 'Extracting ImageNet 64 x 64 tar file...'
            tar.extractall(os.path.join(data_path, 'imagenet_64'))
            tar.close()
            os.remove(os.path.join(data_path, 'imagenet_64', 'valid_64x64.tar'))
            print 'Moving images into directory...'
            os.makedirs(os.path.join(data_path, 'imagenet_64', 'valid_64x64', 'images'))
            for _, _, files in os.walk(os.path.join(data_path, 'imagenet_64', 'valid_64x64')):
                root = os.path.join(data_path, 'imagenet_64', 'valid_64x64')
                for f in files:
                    if os.path.exists(os.path.join(root, f)):
                        os.rename(os.path.join(root, f), os.path.join(root, 'images', f))
        val = os.path.join(data_path, 'imagenet_64', 'valid_64x64')

    else:
        raise Exception('Dataset ' + str(dataset) + ' not found.')

    print 'Data loaded.'
    return (train, val), (train_labels, val_labels), label_names
