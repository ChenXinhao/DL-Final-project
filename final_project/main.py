from util import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from model.pytorch_model import *
from model.tensorflow_model import *
from model.mxnet_model import *

from options.base_options import BaseOptions
import os
import matplotlib.pyplot as plt

from dataset import Dataset

class Dog:
    def __init__(self):
        self.opt = BaseOptions().parse()
        self.dataset = Dataset(self.opt)

    def work(self):
        opt = self.opt

        if opt.model == 'pytorch':
            cur_model = PytorchModel()
        elif opt.model == 'tensorflow':
            cur_model = TensorFlowModel()
        elif opt.model == 'mxnet':
            cur_model = GluonModel()
        else:
            print('what is ' + opt.model)
            exit()

        cur_model.initialize(opt)

        def pass_net(mode, outputFlag):

            if outputFlag:
                pred = np.zeros([0, 527])

            index = 0
            while True:
                data_dict = self.dataset.get_batch_data(index, mode)
                if not data_dict:
                    break
                else:
                    index += 1
                cur_model.set_input(data_dict)
                if mode == 'train':
                    cur_model.train()
                else:
                    cur_model.test()

                # print(cur_model.get_loss())
                if outputFlag:
                    pred = np.concatenate((pred, cur_model.get_output()), axis=0)

            if outputFlag:
                return pred
            else:
                return None

        recorder = {
            'train_mAP': [], 'train_mAUC': [],
            'valid_mAP': [], 'valid_mAUC': [],
        }

        for epoch in range(0, opt.epoch):
            if epoch % opt.output_epoch == 0:
                print('epoch {}:'.format(epoch), end=' ')
                for mode in ['train', 'valid']:
                    tmp_result = pass_net(mode, True)
                    mAP, mAUC = self.dataset.eval_result(tmp_result, mode)
                    recorder[mode + '_mAP'].append(mAP)
                    recorder[mode + '_mAUC'].append(mAUC)
                    print('{0: <3}[mAP:{1:.4f},mAUC:{2:.4f}]'.format(mode, mAP, mAUC), end=' | ')
                print('')
            else:
                pass_net('train', False)

        # save the result
        if opt.eval_mode:
            expr_dir = os.path.join(self.opt.save_dir, self.opt.name)
            file_name = os.path.join(expr_dir, 'output.pkl')
            with open(file_name, 'wb') as f:
                pickle.dump(tmp_result, f)

if __name__ == "__main__":
    dog = Dog()
    dog.work()