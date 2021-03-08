import urllib
import shutil
from os import listdir, makedirs, remove
from os.path import exists, join
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.pcutil import rand_rotation_matrix

all_classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
               'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door',
               'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
               'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
               'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
               'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
               'wardrobe', 'xbox']

number_to_category = {i: c for i, c in enumerate(all_classes)}
category_to_number = {c: i for i, c in enumerate(all_classes)}


class ModelNet40(Dataset):
    def __init__(self, root_dir='/home/datasets/modelnet40', classes=[],
                 transform=[], split='train', valid_percent=10, percent_supervised=0.0):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split (string): `train` or `test`
            valid_percent (int): Percent of train (from the end) to use as valid set.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split.lower()
        self.valid_percent = valid_percent
        self.percent_supervised = percent_supervised

        self._maybe_download_data()

        if self.split in ('train', 'valid'):
            self.files_list = join(self.root_dir, 'train_files.txt')
        elif self.split == 'test':
            self.files_list = join(self.root_dir, 'test_files.txt')
        else:
            raise ValueError('Incorrect split')

        data, labels = self._load_files()

        if classes:
            if classes[0] in all_classes:
                classes = np.asarray([category_to_number[c] for c in classes])
            filter = [label in classes for label in labels]
            data = data[filter]
            labels = labels[filter]
        else:
            classes = np.arange(len(all_classes))

        if self.split in ('train', 'valid'):
            new_data, new_labels = [], []
            if self.percent_supervised > 0.0:
                data_sup, labels_sub = [], []
            for c in classes:
                pc_in_class = sum(labels.flatten() == c)

                if self.split == 'train':
                    portion = slice(0, int(pc_in_class * (1 - (self.valid_percent / 100))))
                else:
                    portion = slice(int(pc_in_class * (1 - (self.valid_percent / 100))), pc_in_class)

                new_data.append(data[labels.flatten() == c][portion])
                new_labels.append(labels[labels.flatten() == c][portion])

                if self.percent_supervised > 0.0:
                    n_max = int(self.percent_supervised * (portion.stop - 1))
                    data_sup.append(data[labels.flatten() == c][:n_max])
                    labels_sub.append(labels[labels.flatten() == c][:n_max])
            data = np.vstack(new_data)
            labels = np.vstack(new_labels)
            if self.percent_supervised > 0.0:
                self.data_sup = np.vstack(data_sup)
                self.labels_sup = np.vstack(labels_sub)
        self.data = data
        self.labels = labels

    def _load_files(self) -> pd.DataFrame:

        with open(self.files_list) as f:
            files = [join(self.root_dir, line.rstrip().rsplit('/', 1)[1]) for line in f]

        data, labels = [], []
        for file in files:
            with h5py.File(file) as f:
                data.extend(f['data'][:])
                labels.extend(f['label'][:])

        return np.asarray(data), np.asarray(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample /= 2  # Scale to [-0.5, 0.5] range
        label = self.labels[idx]

        if 'rotate'.lower() in self.transform:
            r_rotation = rand_rotation_matrix()
            r_rotation[0, 2] = 0
            r_rotation[2, 0] = 0
            r_rotation[1, 2] = 0
            r_rotation[2, 1] = 0
            r_rotation[2, 2] = 1

            sample = sample.dot(r_rotation).astype(np.float32)
        if self.percent_supervised > 0.0:
            id_sup = np.random.randint(self.data_sup.shape[0])
            sample_sup = self.data_sup[id_sup]
            sample_sup /= 2
            label_sup = self.labels_sup[id_sup]
            return sample, label, sample_sup, label_sup
        else:
            return sample, label

    def _maybe_download_data(self):
        if exists(self.root_dir):
            return

        print(f'ModelNet40 doesn\'t exist in root directory {self.root_dir}. '
              f'Downloading...')
        makedirs(self.root_dir)

        url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2][:-5]
        file_path = join(self.root_dir, filename)
        with open(file_path, mode='wb') as f:
            d = data.read()
            f.write(d)

        print('Extracting...')
        with ZipFile(file_path, mode='r') as zip_f:
            zip_f.extractall(self.root_dir)

        remove(file_path)

        extracted_dir = join(self.root_dir, 'modelnet40_ply_hdf5_2048')
        for d in listdir(extracted_dir):
            shutil.move(src=join(extracted_dir, d),
                        dst=self.root_dir)

        shutil.rmtree(extracted_dir)


if __name__ == '__main__':
    ModelNet40()
