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
        # self.label_length = tf.placeholder(dtype=tf.int32, shape=[None],name='label_length')
        # self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        indices = tf.where(tf.not_equal(tf.cast(self.label_data, tf.int32), 0))
        self.label_sparse = tf.SparseTensor(indices=indices, values=tf.gather_nd(self.label_data, indices),
                                            dense_shape=tf.cast(tf.shape(self.label_data), tf.int64))

        self.is_train = tf.placeholder(dtype=tf.bool)

        conv2d_1 = tf.layers.conv2d(self.input_data,32,(3,3),use_bias=True, activation=tf.keras.activations.relu, padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name='conv2d_1')
        tf.summary.scalar( 'conv2d_1', tf.reduce_mean(conv2d_1))
        droput_1 = tf.layers.dropout(conv2d_1,rate=0.1,training=self.is_train)#随机丢失层
        conv2d_2 = tf.layers.conv2d(droput_1, 32, (3, 3), use_bias=True, activation=tf.keras.activations.relu, padding='same',kernel_initializer=tf.keras.initializers.he_normal(),name='conv2d_2')
        tf.summary.scalar('conv2d_2', tf.reduce_mean(conv2d_2))
        max_pool_2 = tf.layers.max_pooling2d(conv2d_2,pool_size=2, strides=2, padding="valid",name='max_pool_2')

        droput_3 = tf.layers.dropout(max_pool_2, rate=0.2,training=self.is_train)  # 随机丢失层
        conv2d_3 = tf.layers.conv2d(droput_3,64, (3,3), use_bias=True, activation=tf.keras.activations.relu, padding='same', kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_3')  # 卷积层
        tf.summary.scalar('conv2d_3', tf.reduce_mean(conv2d_3))
        droput_3 = tf.layers.dropout(conv2d_3, rate=0.2,training=self.is_train)  # 随机丢失层
        conv2d_4 = tf.layers.conv2d(droput_3, 64, (3,3), use_bias=True, activation=tf.keras.activations.relu, padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name= 'conv2d_4')  # 卷积层
        tf.summary.scalar('conv2d_4', tf.reduce_mean(conv2d_4))
        max_pool_4 = tf.layers.max_pooling2d(conv2d_4, pool_size=2, strides=2, padding="valid", name='max_pool_4')

        droput_5 = tf.layers.dropout(max_pool_4, rate=0.3,training=self.is_train)  # 随机丢失层
        conv2d_5 = tf.layers.conv2d(droput_5,128, (3,3), use_bias=True, activation=tf.keras.activations.relu, padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name= 'conv2d_5')  # 卷积层
        tf.summary.scalar('conv2d_5', tf.reduce_mean(conv2d_5))
        droput_6 = tf.layers.dropout(conv2d_5,  rate=0.3,training=self.is_train)  # 随机丢失层
        conv2d_6 = tf.layers.conv2d(droput_6, 128, (3,3), use_bias=True, activation=tf.keras.activations.relu, padding='same', kernel_initializer=tf.keras.initializers.he_normal(),name= 'conv2d_6')  # 卷积层
        tf.summary.scalar('conv2d_6', tf.reduce_mean(conv2d_6))
        max_pool_6 = tf.layers.max_pooling2d(conv2d_6, pool_size=2, strides=2, padding="valid", name='max_pool_6')

        max_pool_shape = max_pool_6.get_shape().as_list()
        max_time,feature,unit = max_pool_shape[1],max_pool_shape[2],max_pool_shape[3]
        output_reshape = tf.reshape(max_pool_6,[-1,max_time,unit*feature])
        droput_7 = tf.layers.dropout(output_reshape, rate=0.4,training=self.is_train)  # 随机丢失层
        all_connect_layer_1 = tf.layers.dense(droput_7, 128, activation=tf.keras.activations.relu, use_bias=True,kernel_initializer=tf.keras.initializers.he_normal(), name='all_connect_layer_1')
        tf.summary.scalar('all_connect_layer_1', tf.reduce_mean(all_connect_layer_1))

        droput_8 = tf.layers.dropout(all_connect_layer_1, rate =0.4,training=self.is_train)  # 随机丢失层
        all_connect_layer_2 = tf.layers.dense(droput_8,self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(),name='all_connect_layer_2')
        tf.summary.scalar('all_connect_layer_2', tf.reduce_mean(all_connect_layer_2))
        # output = tf.reshape(all_connect_layer_2,[-1,max_time,self.MS_OUTPUT_SIZE])

        self.y_predit = tf.keras.activations.softmax(all_connect_layer_2)
        tf.summary.scalar('y_predit', tf.reduce_mean(self.y_predit))


        y_pred = tf.log(tf.transpose(self.y_predit,[1,0,2]) + 1e-7)

        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.label_sparse,y_pred,self.input_length))
        tf.summary.scalar('loss', self.loss)
        self.optimize = tf.train.AdadeltaOptimizer(learning_rate = 0.01, rho = 0.95, epsilon = 1e-06).minimize(self.loss)

        decoded, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(self.y_predit,[1,0,2]), self.input_length, merge_repeated=False)
        # self.predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
        self.accury = tf.edit_distance(tf.cast(decoded[0], tf.int32), self.label_sparse)


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




    def TrainModel(self, datapath, epoch=2, save_step=1000, batch_size=32, filename='model_speech/speech_model24'):
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

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_merge = tf.summary.merge_all()
            train_writter = tf.summary.FileWriter('summary_file',sess.graph)
            for i in range(epoch):
                yielddatas = data.data_genetator(batch_size, self.MAX_TIME)
                pbar = tqdm(yielddatas)
                train_epoch = 0
                train_epoch_size = 500
                for input,_ in pbar:
                    feed = {self.input_data: input[0],self.label_data: input[1],self.input_length:input[2],
                            self.is_train:True}
                    _,loss,train_summary = sess.run([self.optimize,self.loss,summary_merge],feed_dict=feed)
                    train_writter.add_summary(train_summary,train_epoch+i*train_epoch_size)
                    pbar.set_description('epoch:%d/%d,train_epoch: %d/%d ,loss: %s'% (i,epoch,train_epoch_size,train_epoch,loss))
                    if train_epoch == train_epoch_size:
                        break
                    train_epoch +=1
                saver.save(sess, os.path.join(os.getcwd(), 'speech_model_file', 'speech.module'), global_step=i)
                print('测试数据')
                for input,_ in yielddatas:
                    feed = {self.input_data: input[0],self.label_data: input[1],self.input_length:input[2],
                            self.is_train:False}
                    accury = sess.run(self.accury,feed_dict=feed)
                    print('测试数据准确度为： %s' % (accury))
                    break