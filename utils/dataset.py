from cProfile import label
import os
from typing import final
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class IQADatalist():
    def __init__(self, csv_file_path) -> None:
        self.csv_file_path = csv_file_path

    def load_data_dict(self):
        img_r_list, img_a_list, img_b_list, label_list = [], [] ,[], []

        with open(self.csv_file_path, "r") as listFile:
            for line in listFile:
                img_r, img_a, img_b, label = line.split(',')
                label = float(label)

                img_r_list.append(img_r)
                img_a_list.append(img_a)
                img_b_list.append(img_b)
                label_list.append(label)

        label_list = np.array(label_list)
        label_list = label_list.astype('float').reshape(-1, 1)

        data_dict = {
            "img_r_list": img_r_list,
            "img_a_list": img_a_list,
            "img_b_list": img_b_list,
            "label_list": label_list,
        }

        return data_dict


class IQADataset(Dataset):
    def __init__(self, db_path, csv_file_path, transform) -> None:
        super().__init__()

        self.db_path = db_path
        self.csv_file_path = csv_file_path
        self.transform = transform
        self.data_dict = IQADatalist(csv_file_path).load_data_dict()
        self.img_num = len(self.data_dict["img_r_list"])

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):

        img_r_path = self.data_dict['img_r_list'][index]
        img_a_path = self.data_dict['img_a_list'][index]
        img_b_path = self.data_dict['img_b_list'][index]

        img_r = cv2.imread(os.path.join(self.db_path, img_r_path), cv2.IMREAD_COLOR)
        img_a = cv2.imread(os.path.join(self.db_path, img_a_path), cv2.IMREAD_COLOR)
        img_b = cv2.imread(os.path.join(self.db_path, img_b_path), cv2.IMREAD_COLOR)

        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_r = np.array(img_r).astype('float32') / 255
        img_a = np.array(img_a).astype('float32') / 255
        img_b = np.array(img_b).astype('float32') / 255

        r_img = np.transpose(img_r, (2, 0, 1))
        a_img = np.transpose(img_a, (2, 0, 1))
        b_img = np.transpose(img_b, (2, 0, 1))

        label = self.data_dict["label_list"][index]

        sample =  {
            "img_r": r_img,
            "img_a": a_img,
            "img_b": b_img,
            "label": label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
