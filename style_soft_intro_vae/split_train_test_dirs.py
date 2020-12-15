from pathlib import Path
import os
import shutil

if __name__ == '__main__':
    all_data_dir = str(Path.home()) + '/../../mnt/data/tal/celebhq_256'
    train_dir = str(Path.home()) + '/../../mnt/data/tal/celebhq_256_train'
    test_dir = str(Path.home()) + '/../../mnt/data/tal/celebhq_256_test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    num_train = 29000
    num_test = 1000
    images = []
    for i, filename in enumerate(os.listdir(all_data_dir)):
        if i < num_train:
            shutil.copyfile(os.path.join(all_data_dir, filename), os.path.join(train_dir, filename))
        else:
            shutil.copyfile(os.path.join(all_data_dir, filename), os.path.join(test_dir, filename))
