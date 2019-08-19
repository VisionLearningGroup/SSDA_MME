from __future__ import print_function


import argparse
import os
import torch
from model.resnet import resnet34, resnet50
from torch.autograd import Variable
from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_dataset_test

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--step', type=int, default=1000, metavar='step',
                    help='loading step')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='real', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='sketch', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default='multi_all',
                    choices=['multi_all'],
                    help='the name of dataset, multi is large scale dataset')
args = parser.parse_args()
print('dataset %s source %s target %s network %s' %
      (args.dataset, args.source, args.target, args.net))
target_loader_unl, class_list = return_dataset_test(args)
use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet50':
    G = resnet50()
    inc = 2048
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, cosine=True, temp=args.T)
G.cuda()
F1.cuda()
G.load_state_dict(torch.load(os.path.join(args.checkpath,
                                          "G_iter_model_{}_{}_"
                                          "to_{}_step_{}.pth.tar".
                                          format(args.method, args.source,
                                                 args.target, args.step))))
F1.load_state_dict(torch.load(os.path.join(args.checkpath,
                                           "F1_iter_model_{}_{}_"
                                           "to_{}_step_{}.pth.tar".
                                           format(args.method, args.source,
                                                  args.target, args.step))))

im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.cuda()
gt_labels_t = gt_labels_t.cuda()

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def eval(loader, output_file="output.txt"):
    G.eval()
    F1.eval()
    size = 0
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(loader)):
                im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
                paths = data_t[2]
                feat = G(im_data_t)
                output1 = F1(feat)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for i, path in enumerate(paths):
                    f.write("%s %d\n" % (path, pred1[i]))


eval(target_loader_unl, output_file="%s_%s_%s.txt" % (args.method, args.net,
                                                      args.step))
