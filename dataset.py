import torch
from torch.utils.data import Dataset
import json
from utils import transform
import os
from PIL import Image


class mask_dataset(Dataset):
    def __init__(self, dataset, path):
        super(mask_dataset, self).__init__()
        self.path = path
        self.dataset = dataset
        self.image_id = os.listdir(self.path)
        self.image_sizes = list()
        self.items = list()
        for img in self.image_id:
            img_items = img.strip(".jpg").split('__')
            x, y, w, h = json.loads(img_items[1])
            # remove all problematic images
            if (w <= 0 or h <= 0) and dataset == 'train':
                continue
            image = Image.open(os.path.join(self.path, img)).convert('RGB')
            self.image_sizes.append(
                torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0))
            label = [2] if img_items[2] == 'True' else [1]
            bbox = [x, y, w, h]
            bbox = torch.FloatTensor(bbox)
            label = torch.LongTensor(label)
            if self.dataset == 'test':
                image, bbox, label = transform(image, bbox, label, dataset=self.dataset)
            self.items.append((image, bbox, label))

    def __getitem__(self, index):
        img_size = self.image_sizes[index]
        if self.dataset == 'train':
            image, bbox, label = self.items[index]
            image, bbox, label = transform(image, bbox, label, dataset=self.dataset)
            return (image, bbox, label, img_size)
        else:
            image, bbox, label = self.items[index]
            return (image, bbox, label, img_size)

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    data = mask_dataset('train')
    for x in data:
        img, box, label = x
