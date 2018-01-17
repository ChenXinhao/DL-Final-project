import torch
import torch.nn as nn
from .base_model import BaseModel
from .pytorch_networks import *
from torch.autograd import Variable


class PytorchModel(BaseModel):

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
        return self.output_label.data.cpu().numpy()

    def get_loss(self):
        return self.loss.data[0]

    def set_input(self, data_dict):
        self.input_data = Variable(torch.from_numpy(data_dict['data']).float()).cuda()
        self.input_label = Variable(torch.from_numpy(data_dict['label']).float()).cuda()

    def initialize(self, opt):
        self.network = TorchNetwork(opt)
        self.network.cuda()
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.network.parameters()), opt.learn_rate)

    def test(self):
        self.output_label = self.network(self.input_data)

    def train(self):
        self.network.zero_grad()
        self.output_label = self.network(self.input_data)
        self.loss = self.criterion(target=self.input_label, input=self.output_label)
        self.loss.backward()
        self.optimizer.step()
