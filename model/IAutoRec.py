try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import time
import numpy as np
import scipy, math

class IAutoRec():
    def __init__(self, sess, dataset_class, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=500,
                 hidden_neuron=500, verbose=False, T=5, display_step=1000):
        """
        初始化自动编码器推荐系统 (IAutoRec).
        Initialize the Autoencoder for Recommendation Systems (IAutoRec).

        参数 Parameters:
        - sess: TensorFlow session.
        - dataset_class: 数据集类，包含用户、项目信息及评分矩阵。
        - learning_rate: 学习率。
        - reg_rate: 正则化率。
        - epoch: 训练周期数。
        - batch_size: 批次大小。
        - hidden_neuron: 隐藏神经元数量。
        - verbose: 是否打印训练过程信息。
        - T: 未使用的参数。
        - display_step: 显示步骤间隔。
        """

        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.hidden_neuron = hidden_neuron
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.num_users
        self.num_item = dataset_class.num_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()
        self.verbose = verbose
        self.T = T
        self.display_step = display_step

        self.train_data = self.dataset_class.train_matrix.toarray()
        self.train_data_mask = scipy.sign(self.train_data)

        print("IAutoRec.",end=' ')
        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        """
        构建模型网络结构。
        Build the model network structure.
        """
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)
        # Variable
        V = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, self.hidden_neuron], stddev=0.01))
        mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
                            tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        """
        训练模型并返回最后的损失值。
        Train the model and return the last loss value.
        """
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        loss = float('inf')
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={
                                        self.rating_matrix: self.dataset_class.train_matrix[:, batch_set_idx].toarray(),
                                        self.rating_matrix_mask: scipy.sign(
                                            self.dataset_class.train_matrix[:, batch_set_idx].toarray()),
                                        self.keep_rate_net: 1
                                    })  # 0.95
        return loss

    def test(self, test_data):
        """
        对测试数据执行预测并计算RMSE和MAE。
        Perform prediction on test data and calculate RMSE and MAE.
        """
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.reconstruction[u, i]  # self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        """
        执行训练过程并在最后测试评分预测的准确性。
        Execute the training process and finally test the accuracy of rating prediction.
        """
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            if abs(loss_cur - loss_prev) < math.exp(-5):
                break
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset_class.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        """
        将训练好的模型参数保存到指定路径。
        Save the trained model parameters to the specified path.
        """
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        """
        从指定路径加载模型参数。
        Load model parameters from a specified path.
        """
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        """
        预测指定用户对指定项目的评分。
        Predict the rating of a specified user for a specified item.
        """
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        return self.reconstruction[user_id, item_id]