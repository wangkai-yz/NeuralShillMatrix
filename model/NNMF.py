try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import time
import numpy as np
import math

class NNMF():
    def __init__(self, sess, dataset_class, num_factor_1=100, num_factor_2=10, hidden_dimension=50,
                 learning_rate=0.001, reg_rate=0.01, epoch=500, batch_size=256,
                 show_time=False, T=5, display_step=1000):
        """
        初始化神经网络矩阵分解模型 (NNMF).
        Initialize the Neural Network Matrix Factorization (NNMF) model.

        参数 Parameters:
        - sess: TensorFlow session.
        - dataset_class: 数据集类，包含用户、项目信息及评分矩阵。
        - num_factor_1: 第一层因子的数量。
        - num_factor_2: 第二层因子的数量。
        - hidden_dimension: 隐藏层维度。
        - learning_rate: 学习率。
        - reg_rate: 正则化率。
        - epoch: 训练周期数。
        - batch_size: 批次大小。
        - show_time: 是否显示训练时间。
        - T: 显示训练信息的周期。
        - display_step: 显示步骤间隔。
        """
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.num_users
        self.num_item = dataset_class.num_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()

        self.num_factor_1 = num_factor_1
        self.num_factor_2 = num_factor_2
        self.hidden_dimension = hidden_dimension
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("NNMF.")

        self.dataset_class_train_matrix_coo = self.dataset_class.train_matrix.tocoo()
        self.user = self.dataset_class_train_matrix_coo.row.reshape(-1)
        self.item = self.dataset_class_train_matrix_coo.col.reshape(-1)
        self.rating = self.dataset_class_train_matrix_coo.data

        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        """
        构建模型网络结构，包括嵌入层和全连接层。
        Build the model network structure, including embedding layers and fully connected layers.
        """
        print("num_factor_1=%d, num_factor_2=%d, hidden_dimension=%d" % (
            self.num_factor_1, self.num_factor_2, self.hidden_dimension))

        # model dependent arguments
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')
        # latent feature vectors
        P = tf.Variable(tf.random_normal([self.num_user, self.num_factor_1], stddev=0.01))
        Q = tf.Variable(tf.random_normal([self.num_item, self.num_factor_1], stddev=0.01))
        # latent feature matrix(K=1?)
        U = tf.Variable(tf.random_normal([self.num_user, self.num_factor_2], stddev=0.01))
        V = tf.Variable(tf.random_normal([self.num_item, self.num_factor_2], stddev=0.01))

        input = tf.concat(values=[tf.nn.embedding_lookup(P, self.user_id),
                                  tf.nn.embedding_lookup(Q, self.item_id),
                                  tf.multiply(tf.nn.embedding_lookup(U, self.user_id),
                                              tf.nn.embedding_lookup(V, self.item_id))
                                  ], axis=1)
        # tf1->tf2
        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(inputs=input, units=2 * self.num_factor_1 + self.num_factor_2,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=regularizer)
        layer_2 = tf.layers.dense(inputs=layer_1, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_3 = tf.layers.dense(inputs=layer_2, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_4 = tf.layers.dense(inputs=layer_3, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        output = tf.layers.dense(inputs=layer_4, units=1, activation=None,
                                 bias_initializer=tf.random_normal_initializer,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=regularizer)
        self.pred_rating = tf.reshape(output, [-1])
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
                            tf.norm(U) + tf.norm(V) + tf.norm(P) + tf.norm(Q))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        """
        训练模型并返回最后的损失值。
        Train the model and return the last loss value.
        """
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])
        # train
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.y: batch_rating
                                                                            })
        return loss

    def test(self, test_data):
        """
        对测试数据执行预测并计算RMSE和MAE。
        Perform prediction on test data and calculate RMSE and MAE.
        """
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict([u], [i])[0]
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        """
        执行训练过程，并在特定周期打印损失值，最后测试评分预测的准确性。
        Execute the training process, print the loss at specific intervals, and finally test the accuracy of rating prediction.
        """
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            if epoch % self.T == 0:
                print("epoch:\t", epoch, "\tloss:\t", loss_cur)
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
        if type(item_id) != list:
            item_id = [item_id]
        if type(user_id) != list:
            user_id = [user_id] * len(item_id)
        return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]
