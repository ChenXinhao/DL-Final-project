import os
import torch


class BaseModel:
    def __init__(self):
        pass

    def get_output(self):
        pass

    def set_input(self, data_dict):
        pass

    def initialize(self, opt):
        pass

    def test(self):
        pass

    def train(self):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self):
        pass

    # helper loading function that can be used by subclasses
    def load_network(self):
        pass

    # update learning rate (called once every epoch)
    # def update_learning_rate(self):
    #     for scheduler in self.schedulers:
    #         scheduler.step()
    #     lr = self.optimizers[0].param_groups[0]['lr']
    #     print('learning rate = %.7f' % lr)
