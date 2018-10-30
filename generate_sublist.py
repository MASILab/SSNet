import os
import numpy as np
import h5py
import random
import linecache


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def dir2list(path,sub_list_file):
    if os.path.exists(sub_list_file):
        fp = open(sub_list_file, 'r')
        sublines = fp.readlines()
        sub_names = []
        for subline in sublines:
            sub_info = subline.replace('\n', '').split(',')
            sub_names.append(sub_info)
        fp.close()
        return sub_names
    else:
        fp = open(sub_list_file, 'w')
        img_root_dir = os.path.join(path,'img')
        subs = os.listdir(img_root_dir)
        subs.sort()
        sub_names = []
        for sub in subs:
            sub_dir = os.path.join(img_root_dir,sub)
            views = os.listdir(sub_dir)
            views.sort()
            for view in views:
                view_dir = os.path.join(sub_dir,view)
                seg_dir = view_dir.replace('/img/','/seg/')
                slices = os.listdir(view_dir)
                slices.sort()
                for slice in slices:
                    subinfo = (sub,view,slice,view_dir,seg_dir)
                    sub_names.append(subinfo)
                    line = "%s,%s,%s,%s,%s"%(sub,view,slice,view_dir,seg_dir)
                    fp.write(line + "\n")
        fp.close()




def dir2list_folds(path,train_list_file,test_list_file,fold_list_file,fold):

    train_subs,test_subs = get_subs(fold_list_file,fold)

    if os.path.exists(train_list_file) and os.path.exists(test_list_file):
        #train
        fp = open(train_list_file, 'r')
        sublines = fp.readlines()
        train_sub_names = []
        for subline in sublines:
            sub_info = subline.replace('\n', '').split(',')
            train_sub_names.append(sub_info)
        fp.close()
        #test
        fp = open(test_list_file, 'r')
        sublines = fp.readlines()
        test_sub_names = []
        for subline in sublines:
            sub_info = subline.replace('\n', '').split(',')
            test_sub_names.append(sub_info)
        fp.close()
        return train_sub_names,test_sub_names
    else:
        img_root_dir = os.path.join(path,'img')
        #train subjects
        train_sub_names = []
        fp = open(train_list_file, 'w')
        for sub in train_subs:
            sub_dir = os.path.join(img_root_dir,sub)
            views = os.listdir(sub_dir)
            views.sort()
            for view in views:
                view_dir = os.path.join(sub_dir,view)
                seg_dir = view_dir.replace('/img/','/seg/')
                slices = os.listdir(view_dir)
                slices.sort()
                for slice in slices:
                    subinfo = (sub,view,slice,view_dir,seg_dir)
                    train_sub_names.append(subinfo)
                    line = "%s,%s,%s,%s,%s"%(sub,view,slice,view_dir,seg_dir)
                    fp.write(line + "\n")
        fp.close()
        #test subjects
        test_sub_names = []
        fp = open(test_list_file, 'w')
        for sub in test_subs:
            sub_dir = os.path.join(img_root_dir,sub)
            views = os.listdir(sub_dir)
            views.sort()
            for view in views:
                view_dir = os.path.join(sub_dir,view)
                seg_dir = view_dir.replace('/img/','/seg/')
                slices = os.listdir(view_dir)
                slices.sort()
                for slice in slices:
                    subinfo = (sub,view,slice,view_dir,seg_dir)
                    test_sub_names.append(subinfo)
                    line = "%s,%s,%s,%s,%s"%(sub,view,slice,view_dir,seg_dir)
                    fp.write(line + "\n")
        fp.close()
        return train_sub_names,test_sub_names


def get_subs(fold_list_file,fold):
    train_subs = []
    test_subs = []
    fp = open(fold_list_file, 'r')
    sublines = fp.readlines()
    for subline in sublines:
        sub_info = subline.replace('\n', '').split(',')
        if sub_info[0] == 'Names':
            continue
        else:
            grp_ind = sub_info[fold]
            if grp_ind == '0':
                train_subs.append(sub_info[0])
            else:
                test_subs.append(sub_info[0])
    fp.close()
    return train_subs,test_subs