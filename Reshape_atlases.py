import os, sys
import argparse
import scipy.misc
import torch
import numpy as np
import nibabel as nib

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

keys = ['train','test']

for key in keys:
    atlas_img_dir = '/fs4/masi/huoy1/DeepInCyte/Atlases/resample/aladin-reg-images/%s/2labels/' % key
    atlas_seg_dir = '/fs4/masi/huoy1/DeepInCyte/Atlases/resample/aladin-reg-labels/%s/2labels/' % key

    output_img_dir = '/fs4/masi/huoy1/DeepInCyte/Atlases/resample_pythonspace/aladin-reg-images/%s/2labels/' % key
    output_seg_dir = '/fs4/masi/huoy1/DeepInCyte/Atlases/resample_pythonspace/aladin-reg-labels/%s/2labels/' % key


    mkdir(output_img_dir)
    mkdir(output_seg_dir)

    subs = os.listdir(atlas_img_dir)
    for sub in subs:
        atlas_sub_img_file = os.path.join(atlas_img_dir,sub)
        atlas_sub_img = nib.load(atlas_sub_img_file)
        atlas_sub_img_data = atlas_sub_img.get_data()
        output_sub_img_file = os.path.join(output_img_dir,sub)
        output_img = nib.Nifti1Image(atlas_sub_img_data, affine=np.eye(4))
        nib.save(output_img, output_sub_img_file)

    subs = os.listdir(atlas_seg_dir)
    for sub in subs:
        atlas_sub_seg_file = os.path.join(atlas_seg_dir,sub)
        atlas_sub_seg = nib.load(atlas_sub_seg_file)
        atlas_sub_seg_data = atlas_sub_seg.get_data()
        output_sub_seg_file = os.path.join(output_seg_dir,sub)
        output_seg = nib.Nifti1Image(atlas_sub_seg_data, affine=np.eye(4))
        nib.save(output_seg, output_sub_seg_file)