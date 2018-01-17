from .base_model import BaseModel
import mxnet as mx
from mxnet.gluon import data, nn
from mxnet import gluon, autograd, init
import numpy as np
import time


class Net(nn.Block):
    def __init__(self, hidden_size, num_classes):
        super(Net, self).__init__()
        with self.name_scope():
            self.fc1 = nn.Dense(hidden_size)
            self.fc2 = nn.Dense(num_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x))



class GluonModel(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self.network = None
        self.criterion = None
        self.optimizer = None
        self.loss = None
        self.input_data = None
        self.input_label = None
        self.output_label = None

    def get_output(self):
        return self.output_label.asnumpy()

    def get_loss(self):
        return self.loss.asnumpy()

    def set_input(self, data_dict):

        self.input_data = mx.nd.array(data_dict['data'].astype(np.float32)).as_in_context(self.ctx)

        self.input_label = mx.nd.array(data_dict['label'].astype(np.float32)).as_in_context(self.ctx)

    def initialize(self, opt):
        self.net = Net(opt.fc_hidden_size, opt.class_size)
        gpu = True
        self.ctx = mx.gpu() if gpu else mx.cpu()
        self.net.initialize(init=init.Xavier(), ctx=self.ctx)

        self.criterion = gluon.loss.SigmoidBCELoss()
        self.optimizer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': opt.learn_rate})

    def test(self):
        with autograd.record():
            self.output_label = self.net(self.input_data)

    def train(self):
        with autograd.record():
            self.output_label = self.net(self.input_data)
            self.loss = self.criterion(self.output_label, self.input_label)
        self.loss.backward()
        self.optimizer.step(self.input_data.shape[0])
