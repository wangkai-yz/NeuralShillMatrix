try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import math


class GAN_Attacker:
    def __init__(self):
        print("GAN Attack model")

    def DIS(self, input, inputDim, h, activation, hiddenLayers, _reuse=False):
        # input->hidden
        y, _, W, b = self.FullyConnectedLayer(input, inputDim, h, activation, "dis", 0, reuse=_reuse)

        # stacked hidden layers
        for layer in range(hiddenLayers - 1):
            y, _, W, b = self.FullyConnectedLayer(y, h, h, activation, "dis", layer + 1, reuse=_reuse)

        # hidden -> output
        y, _, W, b = self.FullyConnectedLayer(y, h, 1, "none", "dis", hiddenLayers + 1, reuse=_reuse)

        return y

    def GEN(self, input, num_item, h, outputDim, activation, decay, name="gen", _reuse=False):
        """
        input   :   sparse filler vectors
        output  :   reconstructed selected vector
        input: 输入数据。
        num_item: 输入数据的维度。
        h: 隐藏层的大小。
        outputDim: 输出数据的维度。
        activation: 激活函数。
        decay: 衰减系数，用于控制隐藏层的大小。
        name: 层的名称，默认为 "gen"。
        _reuse: 是否重用变量，用于 TensorFlow 的变量共享。

        input: Enter the data
        num_item: The dimension of the input data
        h: Size of the hidden layer.
        outputDim: The dimension of the output data
        activation: The activation function
        decay: This is the attenuation factor used to control the size of the hidden layer.
        name: The name of the layer; defaults to "gen"
        _reuse: Whether to reuse variables for TensorFlow variable sharing.
        """
        # input+thnh
        # 输入+双曲正切函数
        # input_tanh = tf.nn.tanh(input)

        # input->hidden
        y, L2norm, W, b = self.FullyConnectedLayer(input, num_item, h // decay, activation, name, 0, reuse=_reuse)

        # stacked hidden layers
        h = h // decay
        layer = 0
        while True:
            y, this_L2, W, b = self.FullyConnectedLayer(y, h, h // decay, activation, name, layer + 1, reuse=_reuse)
            L2norm = L2norm + this_L2
            layer += 1
            if h // decay > outputDim:
                h = h // decay
            else:
                break

        # hidden -> output
        y, this_L2, W, b = self.FullyConnectedLayer(y, h // decay, outputDim, "none", name, layer + 1, reuse=_reuse)
        L2norm = L2norm + this_L2
        y = tf.nn.sigmoid(y) * 5  # Use the sigmoid activation function and scale the output range to [0, 5]

        return y, L2norm

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

        Create a fully connected layer
        input: The input tensor
        inputDim: The input dimension
        outputDim: The output dimension
        activation: Type of activation function, which can be "none", "sigmoid", "tanh", or "relu"
        model -The name of the model used in variable scope
        layer: The number of the layer
        reuse: Whether to reuse variables (for TensorFlow variable sharing)
        """

        # Calculate the scale of the weight initialization
        scale1 = math.sqrt(6 / (inputDim + outputDim))

        # Variable names that define the weights and biases
        wName = model + "_W" + str(layer)
        bName = model + "_B" + str(layer)

        with tf.variable_scope(model) as scope:
            if reuse == True:
                scope.reuse_variables()

            # Create TensorFlow variables for weights and biases
            W = tf.get_variable(wName, [inputDim, outputDim],initializer=tf.random_uniform_initializer(-scale1, scale1))
            b = tf.get_variable(bName, [outputDim], initializer=tf.random_uniform_initializer(-0.01, 0.01))

            # Compute the output of the fully connected layer
            y = tf.matmul(input, W) + b

            # The L2 regularization term is computed, which is used to regularize the weights
            L2norm = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            # Applying activation functions
            if activation == "none":
                y = tf.identity(y, name="output")
                return y, L2norm, W, b
            elif activation == "sigmoid":
                return tf.nn.sigmoid(y), L2norm, W, b
            elif activation == "tanh":
                return tf.nn.tanh(y), L2norm, W, b
            elif activation == "relu":
                return tf.nn.relu(y), L2norm, W, b
