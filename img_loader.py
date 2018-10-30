import os
import h5py
import numpy as np
from torch.utils import data
import scipy.misc

VGG_MEAN = [103.939, 116.779, 123.68]


def normalizeImage(img):
    img = img.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = img.min()
        maxval = img.max()
        if minval != maxval:
            img -= minval
            img /= (maxval-minval)
    return img*255


class img_loader(data.Dataset):
    def __init__(self, sub_list):
        self.sub_list = sub_list

    def __getitem__(self, index):
        # load image
        subinfo = self.sub_list[index]

        sub_name = subinfo[0]
        view_name = subinfo[1]
        img_name = subinfo[2]
        img_dir = subinfo[3]
        seg_dir = subinfo[4]

        img_file = os.path.join(img_dir, img_name)
        seg_file = os.path.join(seg_dir, img_name)

        gray_img = scipy.misc.imread(img_file)
        gray_img = normalizeImage(gray_img)
        data = np.zeros([3, 512, 512], np.float32)
        data[2, :, :] = gray_img - VGG_MEAN[2]
        data[1, :, :] = gray_img - VGG_MEAN[1]
        data[0, :, :] = gray_img - VGG_MEAN[0]

        seg_img = scipy.misc.imread(seg_file)
        target = np.zeros([512, 512],np.int)
        target[:, :] = seg_img

        target2ch = np.zeros([2, 512, 512],np.float32)
        target2ch[0, :, :] = 1-seg_img
        target2ch[1, :, :] = seg_img
        # target[1, :, :] = seg_img
        return data, target, target2ch, sub_name, view_name, img_name




    def __len__(self):
        self.total_count = len(self.sub_list)
        return self.total_count


    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += np.array(VGG_MEAN)
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl