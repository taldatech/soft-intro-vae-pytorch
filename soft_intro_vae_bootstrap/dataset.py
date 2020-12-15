import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import os


def load_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
               crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False):
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if is_gray is False and img.mode is not 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode is not 'L':
        img = img.convert('L')

    if is_mirror and random.randint(0, 1) is 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            # print([w,cropSize])
            cx1 = random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w - crop_width) / 2.))
            cy2 = cy1 = int(round((h - crop_height) / 2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_width, output_height), Image.BICUBIC)
    return img


class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path,
                 input_height=128, input_width=None, output_height=128, output_width=None,
                 crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):
        super(ImageDatasetFromFile, self).__init__()

        self.image_filenames = image_list
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_gray = is_gray

        self.input_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = load_image(join(self.root_path, self.image_filenames[index]),
                         self.input_height, self.input_width, self.output_height, self.output_width,
                         self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)

        img = self.input_transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)


def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list


class DigitalMonstersDataset(data.Dataset):
    def __init__(self, root_path,
                 input_height=None, input_width=None, output_height=128, output_width=None, is_gray=False, pokemon=True,
                 digimon=True, nexomon=True):
        super(DigitalMonstersDataset, self).__init__()
        image_list = []
        if pokemon:
            print("collecting pokemon...")
            image_list.extend(list_images_in_dir(os.path.join(root_path, 'pokemon')))
        if digimon:
            print("collecting digimon...")
            image_list.extend(list_images_in_dir(os.path.join(root_path, 'digimon', '200')))
        if nexomon:
            print("collecting nexomon...")
            image_list.extend(list_images_in_dir(os.path.join(root_path, 'nexomon')))
        print(f'total images: {len(image_list)}')

        self.image_filenames = image_list
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.is_gray = is_gray

        # self.input_transform = transforms.Compose([
        #     transforms.RandomAffine(0, translate=(5 / output_height, 5 / output_height), fillcolor=(255, 255, 255)),
        #     transforms.ColorJitter(hue=0.5),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        # ])

        self.input_transform = transforms.Compose([
            transforms.RandomAffine(0, translate=(5 / output_height, 5 / output_height), fillcolor=(255, 255, 255)),
            transforms.ColorJitter(hue=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        # self.input_transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])

    def __getitem__(self, index):
        img = load_image(self.image_filenames[index], input_height=self.input_height, input_width=self.input_width,
                         output_height=self.output_height, output_width=self.output_width,
                         crop_height=None, crop_width=None, is_random_crop=False, is_mirror=False, is_gray=False)
        img = self.input_transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)


if __name__ == "__main__":
    ds = DigitalMonstersDataset(root_path='./pokemon_ds')
    print(ds)
