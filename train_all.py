import os, sys
import argparse
import scipy.misc
import torch
import h5py

import numpy as np
import math
import random
from time import time
from torch.autograd import Variable
import torchsrc
import generate_sublist
from img_loader import img_loader



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter(subs,viewName):
	img_subs = []
	for i in range(len(subs)):
		if (subs[i][1]==viewName):
			img_subs.append(subs[i])
	return img_subs


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',default='/fs4/masi/huoy1/DeepInCyte/JournalRevision1/Data2D/2labels', help='the root folder you run experiments')
parser.add_argument('--os_dir',default='/fs4/masi/huoy1/DeepInCyte/JournalRevision1', help='the root folder you run experiments')
parser.add_argument('--network', required=True,type=int,default=0, help='the network that been used')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize_lmk', type=int, default=1, help='input batch size for lmk detection')
parser.add_argument('--batchSize_clss', type=int, default=4, help='input batch size for view classification')
parser.add_argument('--epoch', type=int, default=51, help='number of epochs to train for, default=50')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate, default=0.0001')
parser.add_argument('--compete',type=bool,default=False,help='True: do compete training')
parser.add_argument('--augment',type=bool,default=False,help='True: do use augmented data')
parser.add_argument('--GAN',type=bool,default=False,help='True: add GAN loss')
parser.add_argument('--accreEval',type=bool,default=False,help='True: only evaluate accre results')
parser.add_argument('--viewName',help='viewall | view1 | view2 | view3')
parser.add_argument('--loss_fun',help='Dice | Dice_norm | cross_entropy')
parser.add_argument('--LSGAN',type=bool,default=False,help='True: use LSGAN')
parser.add_argument('--start_epoch',type=int, default=0, help='start epoch')
parser.add_argument('--fold',type=int, default=1, help='fold 1 2 3 4')


opt = parser.parse_args()
print(opt)
input_dir = opt.input_dir
os_dir = opt.os_dir
epoch_num = opt.epoch
lmk_batch_size = opt.batchSize_lmk
clss_batch_size = opt.batchSize_clss
learning_rate = opt.lr
network_num = opt.network
num_workers = 4
compete = opt.compete
augment = opt.augment
GAN = opt.GAN
onlyEval = opt.accreEval
viewName =  opt.viewName
lmk_num = 2
loss_fun = opt.loss_fun
noLSGAN = not opt.LSGAN
start_epoch = opt.start_epoch
fold = opt.fold

# onlyEval = True

fold_str = ('fold%d'%fold)


code_path = os.getcwd()
fcn_torch_code_path = os.path.join(code_path, 'torchfcn')
if fcn_torch_code_path not in sys.path:
     sys.path.insert(0, fcn_torch_code_path)


img_root_dir = input_dir
fold_list_file = os.path.join(os_dir, 'sublist', 'sublist_folds.csv')
train_sublist_name = 'train_list_%s.txt'%fold_str
test_sublist_name = 'test_list_%s.txt'%fold_str


working_root_dir = os.path.join(os_dir, 'ACCRE_revision1')
if os.path.exists('/fs4/masi/huoy1/DeepInCyte/JournalRevision1'):
	sublist_dir = os.path.join(working_root_dir, 'sublist_fs4')
else:
	sublist_dir = os.path.join(working_root_dir,'sublist')
mkdir(sublist_dir)


train_img_list_file = os.path.join(sublist_dir,train_sublist_name)
test_img_list_file = os.path.join(sublist_dir,test_sublist_name)
train_subs,test_subs = generate_sublist.dir2list_folds(img_root_dir,train_img_list_file,test_img_list_file,fold_list_file,fold)



if onlyEval:
	if os.path.exists('/home-local/InCyte_Deep/ACCRE_results/07_17_2017/'):
		working_root_dir = '/home-local/InCyte_Deep/ACCRE_results/07_17_2017/'

if not(viewName == 'viewall'):
	train_subs = filter(train_subs,viewName)
	test_subs = filter(test_subs,viewName)
	if GAN:
		results_path = os.path.join(working_root_dir, 'GAN_results_single')
	else:
		results_path = os.path.join(working_root_dir, 'results_single')
else:
	if GAN:
		results_path = os.path.join(working_root_dir, 'GAN_results')
	else:
		results_path = os.path.join(working_root_dir, 'results')

if opt.LSGAN:
	results_path = str.replace(results_path,'GAN_results','LSGAN_results')

mkdir(results_path)


# test_sub_cate = 'KidneyLong'
# test_sub_name = '0325_I5000003'
# for cc in lmk_test_subs:
# 	if cc[0] == test_sub_cate and cc[3] == test_sub_name:
# 		found_clss_test_sub = cc
# lmk_test_subs = [(found_clss_test_sub)]


# network setting, 1XX clss, 2XX lmk, 3xx MTL
if network_num == 301:  #multi task learning VGG FCN
	model = torchsrc.models.MTL_BN(n_class=11, n_lmk=4, n_networks=5)
	vgg16 = torchsrc.models.VGG16(pretrained=True)
	model.copy_params_from_vgg16(vgg16, copy_classifier=False, copy_fc8=False, init_upscore=False)
elif network_num == 303:  #multi task learning VGG FCN
	model = torchsrc.models.MTL_BN(n_class=11, n_lmk=4, n_networks=5)
	vgg16 = torchsrc.models.VGG16(pretrained=True)
	model.copy_params_from_vgg16(vgg16, copy_classifier=False, copy_fc8=False, init_upscore=False)
elif network_num == 302:
	model = torchsrc.models.MTL_ResNet50(n_classes=11, n_lmks=4, pretrained=True)  # lr = 0.001 or 0.0001
elif network_num == 304:  #compete
	model = torchsrc.models.MTL_ResNet50(n_classes=11, n_lmks=4, pretrained=True)  # lr = 0.001 or 0.0001
elif network_num == 305:
	model = torchsrc.models.MTL_GCN(num_classes=11)  # lr = 0.001 or 0.0001
elif network_num == 101:  #VGG classification
	model = torchsrc.models.ClssNet(n_class=11)  #lr = 0.0001
elif network_num == 102: #ResNet50 classification
	model = torchsrc.models.resnet50(num_classes=11, pretrained=True) #lr = 0.001
elif network_num == 201:
	model = torchsrc.models.FCN32s_BN(n_class=lmk_num)
	vgg16 = torchsrc.models.VGG16(pretrained=True)
	model.copy_params_from_vgg16(vgg16, copy_classifier=False, copy_fc8=False, init_upscore=False)
elif network_num == 202:
	model = torchsrc.models.ResNet50(n_classes=lmk_num,pretrained=True)  #lr = 0.001 or 0.0001
elif network_num == 203:
	model = torchsrc.models.DeconvNet(n_class=lmk_num) #lr = 0.001
elif network_num == 204:
	model = torchsrc.models.fcdensenet56(n_classes=lmk_num) #lr = 0.001
elif network_num == 205:
	model = torchsrc.models.Unet_BN(n_class=lmk_num) #lr = 0.0014
elif network_num == 206:
	model = torchsrc.models.FCNGCN(num_classes=lmk_num)
elif network_num == 207:
	model = torchsrc.models.ResNet101(n_classes=lmk_num, pretrained=True)
elif network_num == 208:
	model = torchsrc.models.ResUnet50(n_classes=lmk_num, pretrained=True)
elif network_num == 501:
	model = torchsrc.models.FCNGCNHuo(num_classes=lmk_num)
elif network_num == 502:
	model = torchsrc.models.ResNetFCN(num_classes=lmk_num)
elif network_num == 503:
	model = torchsrc.models.FCNGCNHuo2D(num_classes=lmk_num)
elif network_num == 504:
	model = torchsrc.models.SSNet(num_classes=lmk_num)
out = os.path.join(results_path,str(network_num),loss_fun,fold_str)
mkdir(out)

train_set = img_loader(train_subs)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=lmk_batch_size,shuffle=True,num_workers=num_workers)
test_set = img_loader(test_subs)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=lmk_batch_size,shuffle=False,num_workers=num_workers)



cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

optim = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))

trainer = torchsrc.Trainer(
	cuda=cuda,
	model=model,
	optimizer=optim,
	train_loader=train_loader,
	test_loader=test_loader,
	out=out,
	network_num = network_num,
	max_epoch = epoch_num,
	compete = compete,
	GAN = GAN,
	batch_size = lmk_batch_size,
	lmk_num = lmk_num,
	onlyEval = onlyEval,
	view = viewName,
	loss_fun = loss_fun,
	noLSGAN = noLSGAN,
)


print("==start training==")
print("==view is == %s "%viewName)

# start_epoch = 0
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train_epoch()

# model_output_file = torch.save(model,os.path.join(out,'model.pth.tar'))


