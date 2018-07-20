import tensorflow as tf
from readdata24 import DataSpeech
from tqdm import tqdm
import os
from tensorflow.python.ops import functional_ops


class Speech_Model():
    def __init__(self):
        '''
        初始化
        默认输出的拼音的表示大小是1422，即1421个拼音+1个空白块
        '''
        self.MAX_TIME=1600      #时间最大长
        self.MAX_FEATURE_LENGTH=200 #特征最大长度
        self.MS_OUTPUT_SIZE=1422    #音素分类

        self.MAX_LABEL_LENGTH = 64  #标签最大长度
        self.lstm_cell_size = 3 #lstm层数
        self.lstm_num_hidden = 256 #lstm隐藏元
        self.get_mode()

    def get_mode(self):
        '''
        		定义CNN/LSTM/CTC模型，使用函数式模型
        		输入层：39维的特征值序列，一条语音数据的最大长度设为1500（大约15s）
        		隐藏层一：1024个神经元的卷积层
        		隐藏层二：池化层，池化窗口大小为2
        		隐藏层三：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
        		隐藏层四：循环层、LSTM层
        		隐藏层五：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
        		隐藏层六：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        		输出层：自定义层，即CTC层，使用CTC的loss作为损失函数，实现连接性时序多输出
        		'''

        self.input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.MAX_TIME,self.MAX_FEATURE_LENGTH,1])
        self.label_data = tf.placeholder(dtype=tf.int32, shape=[None, self.MAX_LABEL_LENGTH])
        self.input_length = tf.placeholder(dtype=tf.int32, shape=[None],name='input_length')
        self.label_length = tf.placeholder(dtype=tf.int32, shape=[None],name='label_length')
        # self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # indices = tf.where(tf.not_equal(tf.cast(self.label_data, tf.int32), 0))
        # self.label_sparse = tf.SparseTensor(indices=indices, values=tf.gather_nd(self.label_data, indices),
        #                                     dense_shape=tf.cast(tf.shape(self.label_data), tf.int64))

        self.is_train = tf.placeholder(dtype=tf.bool)

        conv2d_1 = tf.layers.conv2d(self.input_data,32,(3,3),use_bias=True, padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name='conv2d_1')
        bn_1 = self.batch_norm(conv2d_1,self.is_train, scope='bn_1')
        relu_1 = tf.keras.activations.relu(bn_1)
        # tf.summary.scalar( 'conv2d_1', tf.reduce_mean(relu_1))
        droput_1 = tf.layers.dropout(relu_1,rate=0.1,training=self.is_train)#随机丢失层
        conv2d_2 = tf.layers.conv2d(droput_1, 32, (3, 3), use_bias=True, padding='same',kernel_initializer=tf.keras.initializers.he_normal(),name='conv2d_2')
        bn_2 = self.batch_norm(conv2d_2, self.is_train, scope='bn_2')
        relu_2 = tf.keras.activations.relu(bn_2)
        # tf.summary.scalar('conv2d_2', tf.reduce_mean(relu_2))
        max_pool_2 = tf.layers.max_pooling2d(relu_2,pool_size=2, strides=2, padding="valid",name='max_pool_2')

        droput_3 = tf.layers.dropout(max_pool_2, rate=0.1,training=self.is_train)  # 随机丢失层
        conv2d_3 = tf.layers.conv2d(droput_3,64, (3,3), use_bias=True,  padding='same', kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_3')  # 卷积层
        bn_3 = self.batch_norm(conv2d_3, self.is_train, scope='bn_3')
        relu_3 = tf.keras.activations.relu(bn_3)
        # tf.summary.scalar('conv2d_3', tf.reduce_mean(relu_3))
        droput_3 = tf.layers.dropout(relu_3, rate=0.2,training=self.is_train)  # 随机丢失层
        conv2d_4 = tf.layers.conv2d(droput_3, 64, (3,3), use_bias=True,  padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name= 'conv2d_4')  # 卷积层
        bn_4 = self.batch_norm(conv2d_4, self.is_train, scope='bn_4')
        relu_4 = tf.keras.activations.relu(bn_4)
        # tf.summary.scalar('conv2d_4', tf.reduce_mean(relu_4))
        max_pool_4 = tf.layers.max_pooling2d(relu_4, pool_size=2, strides=2, padding="valid", name='max_pool_4')

        droput_5 = tf.layers.dropout(max_pool_4, rate=0.2,training=self.is_train)  # 随机丢失层
        conv2d_5 = tf.layers.conv2d(droput_5,128, (3,3), use_bias=True, padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name= 'conv2d_5')  # 卷积层
        bn_5 = self.batch_norm(conv2d_5, self.is_train, scope='bn_5')
        relu_5 = tf.keras.activations.relu(bn_5)
        # tf.summary.scalar('conv2d_5', tf.reduce_mean(relu_5))
        droput_6 = tf.layers.dropout(relu_5,  rate=0.3,training=self.is_train)  # 随机丢失层
        conv2d_6 = tf.layers.conv2d(droput_6, 128, (3,3), use_bias=True,  padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name= 'conv2d_6')  # 卷积层
        bn_6 = self.batch_norm(conv2d_6, self.is_train, scope='bn_6')
        relu_6 = tf.keras.activations.relu(bn_6)
        # tf.summary.scalar('conv2d_6', tf.reduce_mean(relu_6))
        max_pool_6 = tf.layers.max_pooling2d(relu_6, pool_size=2, strides=2, padding="valid", name='max_pool_6')

        droput_7 = tf.layers.dropout(max_pool_6, rate=0.3, training=self.is_train)  # 随机丢失层
        conv2d_7 = tf.layers.conv2d(droput_7, 128, (3, 3), use_bias=True,padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name='conv2d_7')  # 卷积层
        bn_7 = self.batch_norm(conv2d_7, self.is_train, scope='bn_7')
        relu_7 = tf.keras.activations.relu(bn_7)
        # tf.summary.scalar('conv2d_7', tf.reduce_mean(relu_7))
        droput_8 = tf.layers.dropout(relu_7, rate=0.4, training=self.is_train)  # 随机丢失层
        conv2d_8 = tf.layers.conv2d(droput_8, 128, (3, 3), use_bias=True, activation=tf.keras.activations.relu,padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name='conv2d_8')  # 卷积层
        bn_8 = self.batch_norm(conv2d_8, self.is_train, scope='bn_8')
        relu_8 = tf.keras.activations.relu(bn_8)
        # tf.summary.scalar('conv2d_8', tf.reduce_mean(relu_8))
        max_pool_8 = tf.layers.max_pooling2d(relu_8, pool_size=1, strides=1, padding="valid", name='max_pool_6')

        max_pool_shape = max_pool_8.get_shape().as_list()
        max_time, feature, unit = max_pool_shape[1], max_pool_shape[2], max_pool_shape[3]
        output_reshape = tf.reshape(max_pool_8, [-1, max_time, unit * feature])

        # forward direction cell
        lstm_fw_cell = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_num_hidden) for _ in range(self.lstm_cell_size) ]
        #backword direction cell
        lstm_bw_cess = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_num_hidden) for _ in range(self.lstm_cell_size) ]

        fbH1,_,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cess,output_reshape,
                                                                dtype=tf.float32,sequence_length=self.input_length)
        fbH1rs = tf.reduce_sum(tf.reshape(fbH1, [-1,max_time, 2, self.lstm_num_hidden]),axis=2)

        droput_9 = tf.layers.dropout(fbH1rs, rate=0.4,training=self.is_train)  # 随机丢失层
        all_connect_layer_1 = tf.layers.dense(droput_9, 128, activation=tf.keras.activations.relu, use_bias=True,kernel_initializer=tf.keras.initializers.he_normal(), name='all_connect_layer_1')
        # tf.summary.scalar('all_connect_layer_1', tf.reduce_mean(all_connect_layer_1))

        droput_10 = tf.layers.dropout(all_connect_layer_1, rate =0.5,training=self.is_train)  # 随机丢失层
        all_connect_layer_2 = tf.layers.dense(droput_10,self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(),name='all_connect_layer_2')
        # tf.summary.scalar('all_connect_layer_2', tf.reduce_mean(all_connect_layer_2))
        # output = tf.reshape(all_connect_layer_2,[-1,max_time,self.MS_OUTPUT_SIZE])

        self.y_predit = tf.keras.activations.softmax(all_connect_layer_2)
        # tf.summary.scalar('y_predit', tf.reduce_mean(self.y_predit))

        sparse_labels = tf.to_int32(self.ctc_label_dense_to_sparse(self.label_data, self.label_length))
        y_pred = tf.log(tf.transpose(self.y_predit,[1,0,2]) + 1e-7)

        self.loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_labels,y_pred,self.input_length))
        tf.summary.scalar('loss', self.loss)

        # global_step = tf.Variable(0, trainable=False)
        # initial_learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.9, staircase=True)
        # tf.summary.scalar('learning_rate', initial_learning_rate)
        self.optimize = tf.train.AdadeltaOptimizer(learning_rate = 0.1, rho = 0.95, epsilon = 1e-06).minimize(self.loss)

        decoded, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(self.y_predit,[1,0,2]), self.input_length, merge_repeated=True)
        self.predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
        self.accury = tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_labels)

    def batch_norm(self,x, phase_train, scope='bn', decay=0.9, eps=1e-5):
        with tf.variable_scope(scope):
            shape = x.get_shape().as_list()
            beta = tf.get_variable(name='beta', shape=[shape[-1]], initializer=tf.constant_initializer(0.0), trainable=True)
            gamma = tf.get_variable(name='gamma', shape=[shape[-1]], initializer=tf.random_normal_initializer(1.0,0.02), trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        return normed

    def ctc_label_dense_to_sparse(self,labels, label_lengths):
        """Converts CTC labels from dense to sparse.

        # Arguments
            labels: dense CTC labels.
            label_lengths: length of the labels.

        # Returns
            A sparse tensor representation of the labels.
        """
        label_shape = tf.shape(labels)
        num_batches_tns = tf.stack([label_shape[0]])
        max_num_labels_tns = tf.stack([label_shape[1]])

        def range_less_than(_, current_input):
            return tf.expand_dims(tf.range(label_shape[1]), 0) < tf.fill(
                max_num_labels_tns, current_input)

        init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
        dense_mask = functional_ops.scan(range_less_than, label_lengths,
                                         initializer=init, parallel_iterations=1)
        dense_mask = dense_mask[:, 0, :]

        label_array = tf.reshape(tf.tile(tf.range(label_shape[1]), num_batches_tns),
                                 label_shape)
        label_ind = tf.boolean_mask(label_array, dense_mask)

        batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(label_shape[0]),
                                                      max_num_labels_tns), self.reverse(label_shape, 0)))
        batch_ind = tf.boolean_mask(batch_array, dense_mask)
        indices = tf.transpose(tf.reshape(self.concatenate([batch_ind, label_ind], axis=0), [2, -1]))

        vals_sparse = tf.gather_nd(labels, indices)

        return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))

    def reverse(self,x, axes):
        """Reverses a tensor along the specified axes.

        # Arguments
            x: Tensor to reverse.
            axes: Integer or iterable of integers.
                Axes to reverse.

        # Returns
            A tensor.
        """
        if isinstance(axes, int):
            axes = [axes]
        return tf.reverse(x, axes)

    def concatenate(self,tensors, axis=-1):
        """Concatenates a list of tensors alongside the specified axis.

        # Arguments
            tensors: list of tensors to concatenate.
            axis: concatenation axis.

        # Returns
            A tensor.
        """
        if axis < 0:
            rank = self.ndim(tensors[0])
            if rank:
                axis %= rank
            else:
                axis = 0

        if all([isinstance(x, tf.SparseTensor) for x in tensors]):
            return tf.sparse_concat(axis, tensors)
        else:
            return tf.concat([self.to_dense(x) for x in tensors], axis)

    def to_dense(self,tensor):
        """Converts a sparse tensor into a dense tensor and returns it.

        # Arguments
            tensor: A tensor instance (potentially sparse).

        # Returns
            A dense tensor.

        # Examples
        ```python
            >>> from keras import backend as K
            >>> b = K.placeholder((2, 2), sparse=True)
            >>> print(K.is_sparse(b))
            True
            >>> c = K.to_dense(b)
            >>> print(K.is_sparse(c))
            False
        ```
        """
        if isinstance(tensor, tf.SparseTensor):
            return tf.sparse_tensor_to_dense(tensor)
        else:
            return tensor

    def ndim(self,x):
        """Returns the number of axes in a tensor, as an integer.

        # Arguments
            x: Tensor or variable.

        # Returns
            Integer (scalar), number of axes.

        # Examples
        ```python
            >>> from keras import backend as K
            >>> inputs = K.placeholder(shape=(2, 4, 5))
            >>> val = np.array([[1, 2], [3, 4]])
            >>> kvar = K.variable(value=val)
            >>> K.ndim(inputs)
            3
            >>> K.ndim(kvar)
            2
        ```
        """
        dims = x.get_shape()._dims
        if dims is not None:
            return len(dims)
        return None

    # def conv_2d(self,input,shape,name):
    #     with tf.variable_scope(name) as scope:
    #         weights = tf.Variable(name='weights',initial_value=tf.truncated_normal(shape, stddev=0.1))
    #         tf.summary.scalar(name + '_weights', tf.reduce_mean(weights))
    #         biases = tf.Variable(name='biases',initial_value=tf.constant(0.1, shape=[shape[-1]]))
    #         tf.summary.scalar(name + '_biases', tf.reduce_mean(biases))
    #         conv = tf.nn.bias_add(tf.nn.conv2d(input,weights,[1,1,1,1],padding='SAME'),biases)
    #         relu = tf.nn.relu(conv)
    #         return relu
    #
    # def all_connect_layer(self,input,units,name,is_activate=True):
    #     with tf.variable_scope(name):
    #         shape = input.get_shape().as_list()
    #         weight = tf.Variable(tf.truncated_normal([shape[-1],units], stddev=0.1))
    #         tf.summary.scalar(name+'_weight', tf.reduce_mean(weight))
    #         bias = tf.Variable(tf.constant(0.1, shape=[units]))
    #         tf.summary.scalar(name+'_bias', tf.reduce_mean(bias))
    #         output = tf.nn.bias_add(tf.matmul(input, weight),bias)
    #         if is_activate:
    #             output= tf.nn.relu(output)
    #         return output




    def TrainModel(self, datapath, epoch=2, save_step=1000, batch_size=32):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        data = DataSpeech(datapath, 'train')

        # num_data = data.GetDataNum()  # 获取数据的数量
        txt_loss = open(
            os.path.join(os.getcwd(), 'speech_log_file', 'Test_Report_loss.txt'),
            mode='a', encoding='UTF-8')

        txt_obj = open(
            os.path.join(os.getcwd(), 'speech_log_file', 'Test_Report_accuracy.txt'),
            mode='a', encoding='UTF-8')

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess,os.path.join(os.getcwd(), 'speech_model_file','speech.module-3'))
            summary_merge = tf.summary.merge_all()
            train_writter = tf.summary.FileWriter('summary_file',sess.graph)
            for i in range(4,epoch):
                yielddatas = data.data_genetator(batch_size, self.MAX_TIME)
                pbar = tqdm(yielddatas)
                train_epoch = 0
                train_epoch_size = save_step
                for input,_ in pbar:
                    feed = {self.input_data: input[0],self.label_data: input[1],self.input_length:input[2],self.label_length:input[3],
                            self.is_train:True}
                    _,loss,train_summary = sess.run([self.optimize,self.loss,summary_merge],feed_dict=feed)
                    train_writter.add_summary(train_summary,train_epoch+i*train_epoch_size)
                    pr = 'epoch:%d/%d,train_epoch: %d/%d ,loss: %s'% (epoch,i,train_epoch_size,train_epoch,loss)
                    pbar.set_description(pr)
                    txt = pr + '\n'
                    txt_loss.write(txt)
                    if train_epoch == train_epoch_size:
                        break
                    train_epoch +=1
                    if train_epoch%3000==0:
                        self.TestMode(data, sess, i,txt_obj)
                saver.save(sess, os.path.join(os.getcwd(), 'speech_model_file', 'speech.module'), global_step=i)
            txt_loss.close()


    def TestMode(self,data,sess,epoch,txt_obj):
        import time
        # 测试数据集
        nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


        edit_sum = 0
        for i in range(0,5):
            test_data, _ = data.get_data(20, self.MAX_TIME, ran_num=1000+i*20)
            feed = {self.input_data: test_data[0], self.label_data: test_data[1], self.input_length: test_data[2],
                    self.label_length: test_data[3],
                    self.is_train: False}
            accury,pre = sess.run([self.accury,self.predict], feed_dict=feed)

            for e in accury:
                if e > 1:
                    e = 1
                edit_sum += e
            txt = ''
            txt += str(i) + '\n'
            txt += 'True:\t' + str(test_data[1]) + '\n'
            txt += 'Pred:\t' + str(pre) + '\n'
            txt += '\n'
            txt_obj.write(txt)
            # txt_obj.write(txt)
        # p_str = ''
        # for p in pre[0]:
        #     p_str += data.list_symbol[p]
        # la_str = ''
        # for td in test_data[1][0]:
        #     if td != 0:
        #         la_str += data.list_symbol[td]
        # print('\n测试第一条标签数据为：'+la_str)
        # print('测试第一条预测数据为：'+p_str)
        error_rate = edit_sum / 100 * 100
        print('测试数据错误率为： %s ' % (error_rate) + '%')
        txt = ''
        txt += '测试数据错误率为： %s ' % (error_rate) + '%'
        txt += '\n'
        txt_obj.write(txt)