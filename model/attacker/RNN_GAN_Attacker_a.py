
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import math


class RNN_GAN_Attacker_yan:
    def __init__(self):
        print("RNN GAN Attack model")

    def DIS(self, input, inputDim, h, activation, hiddenLayers, _reuse=False):

        with tf.variable_scope("D_rnn", reuse=_reuse):
            cell = tf.nn.rnn_cell.BasicRNNCell(64)
            input_3d = tf.expand_dims(input, axis=-1)
            outputs, state = tf.nn.dynamic_rnn(cell, input_3d, dtype=tf.float32)
        
        # print(outputs.shape)
        outputs = tf.reshape(outputs, [-1, inputDim*64])
        
        # input->hidden
        y, _, W, b = self.FullyConnectedLayer(outputs, inputDim*64, h, activation, "dis", 0, reuse=_reuse)


        # hidden -> output
        y, _, W, b = self.FullyConnectedLayer(y, h, 1, "none", "dis", hiddenLayers + 1, reuse=_reuse)

        return y

    def GEN(self, input, num_item, h, outputDim, activation, decay, name="gen", _reuse=False):
        
        """
        input   :   sparse filler vectors
        output  :   reconstructed selected vector
        """
        # input+thnh
        # input_tanh = tf.nn.tanh(input)

        # input->hidden
        #RNN 
        print('input',input.shape)
        # """输入数据为(batch_size, time_step, input_size)
        with tf.variable_scope(name + "_rnn", reuse=_reuse):
            cell = tf.nn.rnn_cell.BasicRNNCell(64)
            input_3d = tf.expand_dims(input, axis=-1)
            outputs, state = tf.nn.dynamic_rnn(cell, input_3d, dtype=tf.float32)
        
        # print(outputs.shape)
        outputs = tf.reshape(outputs, [-1, num_item*64])
        # print(outputs.shape)
        

        y, L2norm, W, b = self.FullyConnectedLayer(outputs, num_item*64, h // decay, activation, name, 0, reuse=_reuse)
        # print('y',y.shape)
        
        # hidden -> output
        y, this_L2, W, b = self.FullyConnectedLayer(y, h // decay, outputDim, "none", name, 1, reuse=_reuse)
        L2norm = L2norm + this_L2
        y = tf.nn.sigmoid(y) * 5
        # print(f'end_y',y.shape)
        # sdf
        return y, L2norm

    def FullyConnectedLayer(self, input, inputDim, outputDim, activation, model, layer, reuse=False):
        scale1 = math.sqrt(6 / (inputDim + outputDim))

        wName = model + "_W" + str(layer)
        bName = model + "_B" + str(layer)

        with tf.variable_scope(model) as scope:

            if reuse == True:
                scope.reuse_variables()

            W = tf.get_variable(wName, [inputDim, outputDim],
                                initializer=tf.random_uniform_initializer(-scale1, scale1))
            b = tf.get_variable(bName, [outputDim], initializer=tf.random_uniform_initializer(-0.01, 0.01))

            y = tf.matmul(input, W) + b

            L2norm = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            if activation == "none":
                y = tf.identity(y, name="output")
                return y, L2norm, W, b

            elif activation == "sigmoid":
                return tf.nn.sigmoid(y), L2norm, W, b

            elif activation == "tanh":
                return tf.nn.tanh(y), L2norm, W, b
            elif activation == "relu":
                return tf.nn.relu(y), L2norm, W, b
            
