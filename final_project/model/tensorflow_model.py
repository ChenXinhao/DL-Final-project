from .base_model import BaseModel
import tensorflow as tf
import numpy as np


class TensorFlowModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.network = None
        self.criterion = None
        self.optimizer = None
        self.loss = None
        self.input_data = None
        self.input_label = None
        self.output_label = None

        self.tf_input_x = None
        self.tf_input_y = None
        self.tf_pred = None
        self.tf_loss = None
        self.tf_optimizer = None

    def get_output(self):
        return self.output_label

    def get_loss(self):
        return self.loss

    def set_input(self, data_dict):
        self.input_data = data_dict['data'].astype(np.float)
        self.input_label = data_dict['label'].astype(np.float)

    def initialize(self, opt):
        self.criterion = tf.nn.sigmoid_cross_entropy_with_logits
        self.optimizer = tf.train.AdamOptimizer
        graph = tf.Graph()
        # 'xavier'
        initializer = tf.contrib.layers.xavier_initializer()

        # build graph
        with graph.as_default():
            with tf.name_scope('Inputs'):
                self.tf_input_x = tf.placeholder(tf.float32, shape=[None, opt.utterance_size, opt.feature_size])
                tmp_x = tf.reshape(self.tf_input_x, [-1, opt.utterance_size * opt.feature_size])
                self.tf_input_y = tf.placeholder(tf.float32, shape=[None, opt.class_size])

            with tf.name_scope('Network'):
                tmp_x = tf.layers.dense(tmp_x, opt.fc_hidden_size,
                                        bias_initializer=initializer,
                                        kernel_initializer=initializer)
                self.tf_pred = tf.layers.dense(tmp_x, opt.class_size,
                                        bias_initializer=initializer,
                                        kernel_initializer=initializer)

            with tf.name_scope('Loss'):
                self.tf_loss = self.criterion(labels=self.tf_input_y, logits=self.tf_pred)

            with tf.name_scope('Optimizer'):
                self.tf_optimizer = self.optimizer(learning_rate=opt.learn_rate).minimize(self.tf_loss)

            init = tf.global_variables_initializer()

        # initialize
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)
        sess.run(init)

        self.network = sess

    def test(self):
        feed_dict = {
            self.tf_input_x: self.input_data,
            self.tf_input_y: self.input_label
        }
        self.output_label, _ = self.network.run([self.tf_pred, self.tf_loss], feed_dict=feed_dict)

    def train(self):
        feed_dict = {
            self.tf_input_x: self.input_data,
            self.tf_input_y: self.input_label
        }

        self.output_label, self.loss, _ = self.network.run([self.tf_pred, self.tf_loss, self.tf_optimizer], feed_dict=feed_dict)
