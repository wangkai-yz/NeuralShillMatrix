try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import math


class RNN_GAN_Attacker:
    def __init__(self):
        print("RNN-GAN Attack model")

    def RNN_Layer(self, input, hidden_units, activation, scope_name, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            # 定义RNN单元，这里使用基础的LSTM单元
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)

            # 通过dynamic_rnn进行RNN计算
            outputs, _ = tf.nn.dynamic_rnn(rnn_cell, input, dtype=tf.float32)

            # 应用激活函数
            if activation == "tanh":
                outputs = tf.nn.tanh(outputs)
            elif activation == "relu":
                outputs = tf.nn.relu(outputs)

            return outputs

    def GEN(self, input, hidden_units, output_dim, activation, name="gen", reuse=False):
        """
        使用RNN作为生成器
        input: 输入张量
        hidden_units: RNN层的隐藏单元数量
        output_dim: 输出维度
        activation: 激活函数
        name: 层的名称
        reuse: 是否重用变量
        """
        input_expanded = tf.expand_dims(input, 1)
        rnn_output = self.RNN_Layer(input_expanded, hidden_units, activation, name, reuse=reuse)

        # 取RNN的最后一步输出
        last_output = rnn_output[:, -1, :]

        # Fully connected layer to shape the output
        y, L2norm, W, b = self.FullyConnectedLayer(last_output, hidden_units, output_dim, "none", name, 0, reuse=reuse)
        y = tf.nn.sigmoid(y) * 5

        return y, L2norm

    def DIS(self, input, input_dim, hidden_units, activation, hidden_layers, _reuse=False):

        y, _, W, b = self.FullyConnectedLayer(input, input_dim, hidden_units, activation, "dis", 0, reuse=_reuse)

        for layer in range(hidden_layers - 1):
            y, _, W, b = self.FullyConnectedLayer(y, hidden_units, hidden_units, activation, "dis", layer + 1,reuse=_reuse)

        y, _, W, b = self.FullyConnectedLayer(y, hidden_units, 1, "none", "dis", hidden_layers + 1, reuse=_reuse)
        return y

    def FullyConnectedLayer(self, input, inputDim, outputDim, activation, model, layer, reuse=False):
        """
        创建一个全连接层
        input: 输入张量
        inputDim: 输入维度
        outputDim: 输出维度
        activation: 激活函数类型，可以是 "none"（无激活函数）、"sigmoid"、"tanh" 或 "relu"
        model: 模型名称，用于变量作用域
        layer: 层的编号
        reuse: 是否重用变量（用于 TensorFlow 的变量共享）
        """

        # 计算权重初始化的标度
        scale1 = math.sqrt(6 / (inputDim + outputDim))

        # 定义权重和偏差的变量名称
        wName = model + "_W" + str(layer)
        bName = model + "_B" + str(layer)

        with tf.variable_scope(model) as scope:
            if reuse == True:
                scope.reuse_variables()

            # 创建权重和偏差的 TensorFlow 变量
            W = tf.get_variable(wName, [inputDim, outputDim],initializer=tf.random_uniform_initializer(-scale1, scale1))
            b = tf.get_variable(bName, [outputDim], initializer=tf.random_uniform_initializer(-0.01, 0.01))

            # 计算全连接层的输出
            y = tf.matmul(input, W) + b

            # 计算 L2 正则化项，用于正则化权重
            L2norm = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            # 应用激活函数
            if activation == "none":
                y = tf.identity(y, name="output")
                return y, L2norm, W, b
            elif activation == "sigmoid":
                return tf.nn.sigmoid(y), L2norm, W, b
            elif activation == "tanh":
                return tf.nn.tanh(y), L2norm, W, b
            elif activation == "relu":
                return tf.nn.relu(y), L2norm, W, b



