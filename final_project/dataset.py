from util import*

class Dataset:

    def __init__(self, opt):

        def padding(datas):
            new_datas = []
            for data in datas:
                new_data = np.concatenate((data, np.zeros([10 - data.shape[0], 128])), axis=0)
                new_datas.append(new_data)
            new_datas = np.array(new_datas, dtype=float)
            new_datas /= 255.0
            return new_datas

        self.opt = opt
        if opt.use_big_train:
            train_data_dict = read_dir('./big_train')
        else:
            train_data_dict = read_dir('./train')

        self.train_data, self.train_label = format_data(train_data_dict)
        self.train_data = padding(self.train_data)

        if opt.eval_mode:
            valid_data_dict = read_dir('./eval')
            self.valid_data, self.valid_label = format_data(valid_data_dict)
            self.valid_data = padding(self.valid_data)
        else:
            data_len = self.train_data.shape[0]
            cut_position = int(data_len * 0.2)
            self.valid_data, self.valid_label = self.train_data[0:cut_position, ...], self.train_label[0:cut_position, ...]
            self.train_data, self.train_label = self.train_data[cut_position:data_len, ...], self.train_label[cut_position:data_len, ...]

        print('train data size: {}, valid data size {}'.format(self.train_data.shape[0], self.valid_data.shape[0]))

    def get_batch_data(self, index, mode):
        if mode == 'train':
            data = self.train_data
            label = self.train_label
        elif mode == 'valid':
            data = self.valid_data
            label = self.valid_label
        else:
            print('what is ' + mode + ' ?')
            exit()
            
        batch_size = self.opt.batch_size
        start = index * batch_size
        limit_size = len(data)
        if(start >= limit_size):
            return None
        end = min(limit_size, (1 + index) * batch_size)

        data = {
            'data': data[start: end],
            'label': label[start: end],
        }
        return data

    def eval_result(self, pred, mode):
        if mode == 'train':
            return eval(pred, self.train_label)
        elif mode == 'valid':
            return eval(pred, self.valid_label)
        else:
            print('what is ' + mode + ' ?')
            exit()
