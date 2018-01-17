import argparse
import os
import torch
import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.opt=None

    def initialize(self):
        # environment setting
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment.')
        self.parser.add_argument('--save_dir', type=str, default='./result', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pytorch', help='pytorch tensorflow mxnet')
        # self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # data setting
        self.parser.add_argument('--skip_lstm', type=int, default=0)
        self.parser.add_argument('--utterance_size', type=int, default=10)
        self.parser.add_argument('--feature_size', type=int, default=128)
        self.parser.add_argument('--class_size', type=int, default=527)
        self.parser.add_argument('--use_big_train', type=bool, default=False)
        self.parser.add_argument('--eval_mode', type=bool, default=False, help='use eval data')

        # model setting
        self.parser.add_argument('--fc_hidden_size', type=int, default=500)
        self.parser.add_argument('--init_type', type=str, default='xavier', help='normal xavier kaiming orthogonal')
        self.parser.add_argument('--drop_out', type=float, default=0.0)

        # training setting self.parser.add_argument('--fc_hidden_size', type=int, default=1000)
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--output_epoch', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=500)
        self.parser.add_argument('--optim_func', type=str, default='adam')
        self.parser.add_argument('--learn_rate', type=float, default=0.001)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.save_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
