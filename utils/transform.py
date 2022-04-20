import torch
import numpy as np


class RandCrop(object):
    def __init__(self, output_size, crop_time=3):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.crop_time = crop_time

    def __call__(self, sample):
        # img_r : C x H x W (numpy)
        img_r, img_a, img_b = sample['img_r'], sample['img_a'], sample['img_b']
        label = sample['label']

        c, h, w = img_r.shape
        new_h, new_w = self.output_size

        final_r, final_a, final_b = None, None, None

        for i in range(self.crop_time):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            tmp_r = img_r[:, top: top+new_h, left: left+new_w]
            tmp_a = img_a[:, top: top+new_h, left: left+new_w]
            tmp_b = img_b[:, top: top+new_h, left: left+new_w]

            if i == 0:
                final_r = tmp_r
                final_a = tmp_a
                final_b = tmp_b
            else:
                final_r = np.concatenate((final_r, tmp_r), axis=0)
                final_a = np.concatenate((final_a, tmp_a), axis=0)
                final_b = np.concatenate((final_b, tmp_b), axis=0)

        sample = {
            "img_r": final_r,
            "img_a": final_a,
            "img_b": final_b,
            "label": label,
        }

        return sample


class RandHorizontalFlip(object):
    def __call__(self, sample):
        # img_r: C x H x W (numpy)
        img_r, img_a, img_b = sample['img_r'], sample['img_a'], sample['img_b']
        label = sample['label']

        prob_lr = np.random.random()
        # np.fliplr needs HxWxC -> transpose from CxHxW to HxWxC
        # after the flip ends, return to CxHxW
        if prob_lr > 0.5:
            img_r = np.fliplr(img_r.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
            img_a = np.fliplr(img_a.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
            img_b = np.fliplr(img_b.transpose((1, 2, 0))).copy().transpose((2, 0, 1))

        sample = {'img_r': img_r, 'img_a': img_a, 'img_b': img_b, 'label': label}
        return sample


class RandRotation(object):
    def __call__(self, sample):
        # img_r: C x H x W (numpy)
        img_r, img_a, img_b = sample['img_r'], sample['img_a'], sample['img_b']
        label = sample['label']

        prob_rot = np.random.uniform()

        if prob_rot < 0.25:     # rot0
            pass
        elif prob_rot < 0.5:    # rot90
            img_r = np.rot90(img_r.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
            img_a = np.rot90(img_a.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
            img_b = np.rot90(img_b.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
        elif prob_rot < 0.75:   # rot180
            img_r = np.rot90(img_r.transpose((1, 2, 0)), 2).copy().transpose((2, 0, 1))
            img_a = np.rot90(img_a.transpose((1, 2, 0)), 2).copy().transpose((2, 0, 1))
            img_b = np.rot90(img_b.transpose((1, 2, 0)), 2).copy().transpose((2, 0, 1))
        else:                   # rot270
            img_r = np.rot90(img_r.transpose((1, 2, 0)), 3).copy().transpose((2, 0, 1))
            img_a = np.rot90(img_a.transpose((1, 2, 0)), 3).copy().transpose((2, 0, 1))
            img_b = np.rot90(img_b.transpose((1, 2, 0)), 3).copy().transpose((2, 0, 1))

        sample = {'img_r': img_r, 'img_a': img_a, 'img_b': img_b, 'label': label}
        return sample



class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # img_r: C x H x W (numpy)
        img_r, img_a, img_b = sample['img_r'], sample['img_a'], sample['img_b']
        label = sample['label']

        img_r = (img_r - self.mean) / self.var
        img_a = (img_a - self.mean) / self.var
        img_b = (img_b - self.mean) / self.var

        sample = {'img_r': img_r, 'img_a': img_a, 'img_b': img_b, 'label': label}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # img_r: C x H x W (numpy->tensor)
        img_r, img_a, img_b = sample['img_r'], sample['img_a'], sample['img_b']
        label = sample['label']

        img_r = torch.from_numpy(img_r)
        img_a = torch.from_numpy(img_a)
        img_b = torch.from_numpy(img_b)
        label = torch.from_numpy(label)

        sample = {'img_r': img_r, 'img_a': img_a, 'img_b': img_b, 'label': label}
        return sample


def RandShuffle(scenes, train_size=0.8):
    if scenes == "all":
        scenes = list(range(200))

    n_scenes = len(scenes)
    n_train_scenes = int(np.floor(n_scenes * train_size))
    n_test_scenes = n_scenes - n_train_scenes

    seed = np.random.random()
    random_seed = int(seed*10)
    np.random.seed(random_seed)
    np.random.shuffle(scenes)
    train_scene_list = scenes[:n_train_scenes]
    test_scene_list = scenes[n_train_scenes:]

    return train_scene_list, test_scene_list
