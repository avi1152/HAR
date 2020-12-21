import torch
import cv2
import numpy as np
import numbers
import collections
import random


class ComposeMix(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            if t[1] == "img":
                for idx, img in enumerate(imgs):
                    imgs[idx] = t[0](img)
            elif t[1] == "vid":
                imgs = t[0](imgs)
            else:
                print("Please specify the transform type")
                raise ValueError
        return imgs


class RandomCropVideo(object):

    def __init__(self, size, padding=0, pad_method=cv2.BORDER_CONSTANT):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_method = pad_method

    def __call__(self, imgs):
        th, tw = self.size
        h, w = imgs[0].shape[:2]
        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        for idx, img in enumerate(imgs):
            if self.padding > 0:
                img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                         self.padding, self.padding,
                                         self.pad_method)
            # sample crop locations if not given
            # it is necessary to keep cropping same in a video
            img_crop = img[y1:y1 + th, x1:x1 + tw]
            imgs[idx] = img_crop
        return imgs


class RandomHorizontalFlipVideo(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            for idx, img in enumerate(imgs):
                imgs[idx] = cv2.flip(img, 1)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomReverseTimeVideo(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            imgs = imgs[::-1]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotationVideo(object):

    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, imgs):
        h, w = imgs[0].shape[:2]
        degree_sampled = np.random.choice(
                            np.arange(-self.degree, self.degree, 0.5))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degree_sampled, 1)

        for idx, img in enumerate(imgs):
            imgs[idx] = cv2.warpAffine(img, M, (w, h))

        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(degree={})'.format(self.degree_sampled)


class IdentityTransform(object):
    def __init__(self,):
        pass

    def __call__(self, imgs):
        return imgs


class Scale(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                if ow < w:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if oh < h:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
        else:
            return cv2.resize(img, tuple(self.size))

