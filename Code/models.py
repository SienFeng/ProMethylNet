import tensorflow as tf
from spektral.layers import GATConv
from tensorflow.keras.initializers import GlorotUniform
from utils import f1_m, positive_accuracy_m, negative_accuracy_m, \
    precision_m, recall_m, accuracy_m, weighted_binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

initializer = GlorotUniform(seed=42)

#  多尺度卷积神经网络模型，用于提取局部特征
class MultiCNN(tf.keras.Model):
    def __init__(self, input_dim, dropout_rate):
        super(MultiCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(1e-4)
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(1e-4)
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=7,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(1e-4)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)

    def call(self, inputs):
        # 多尺度卷积
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        # 特征拼接
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.dropout(x)
        x = self.batch_norm(x)
        # 自注意力机制
        x = self.attention(x, x)
        return x

# 自定义的 GATConv 层，用于处理输入掩码问题
class GATConvNoMask(GATConv):
    def __init__(self, *args, **kwargs):
        super(GATConvNoMask, self).__init__(*args, **kwargs)

    def call(self, inputs):
        # 移除mask参数，因为父类的call方法不接受这个参数
        return super(GATConvNoMask, self).call(inputs)

# 构建基于图注意力网络的模型
def build_graphphos_model(in_channels, hidden_channels, num_layers, num_heads=4, dropout_rate=0.3):
    # 输入层，序列长度为可变的 None
    x_in = tf.keras.Input(shape=(None, in_channels))
    a_in = tf.keras.Input(shape=(None, None))
    x = x_in
    # 循环构建4层
    for _ in range(num_layers):
        x = GATConvNoMask(
            channels=hidden_channels,
            activation='relu',
            attn_heads=num_heads,
            kernel_regularizer=l2(1e-4),
            bias_regularizer=l2(1e-4)
        )([x, a_in])
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    model = tf.keras.Model(inputs=[x_in, a_in], outputs=x)
    return model

# 定义FNN
class MultiFNN(tf.keras.Model):
    def __init__(self, hidden_units, dropout_rate):
        super(MultiFNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            kernel_regularizer=l2(1e-4)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='sigmoid')
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.batch_norm(x)
        output = self.dense2(x)
        return output

# 构建模型
def build_model(hp_unused, in_channels, seq_len, class_weights, model_params=None):

    with tf.device('/GPU:0'):
        hidden_channels = 640
        num_layers = 4
        learning_rate = 1e-4
        dropout_rate = 0.3

        # 消融的开关，看哪些需要打开
        use_cnn = True
        use_gat = True
        use_attention = True
        use_lstm = True
        if model_params is not None:
            use_cnn = model_params.get('use_cnn', True)
            use_gat = model_params.get('use_gat', True)
            use_attention = model_params.get('use_attention', True)
            use_lstm = model_params.get('use_lstm', True)

        # 输入层，序列，邻接矩阵，掩码
        input_layer = tf.keras.Input(shape=(None, in_channels), name='input_layer')
        adjacency_input = tf.keras.Input(shape=(None, None), name='adjacency_input')
        mask_input = tf.keras.Input(shape=(None,), name='mask_input', dtype=tf.bool)

        # 添加Masking层，忽略填充值
        masked_input = tf.keras.layers.Masking(mask_value=0.0)(input_layer)

        # 如果用CNN
        if use_cnn:
            cnn_model = MultiCNN(input_dim=in_channels, dropout_rate=dropout_rate)
            cnn_output = cnn_model(masked_input)
        else:
            # 不使用CNN时，cnn_output就直接等于masked_input
            cnn_output = masked_input

        # 如果使用GAT
        if use_gat:
            gnn_model = build_graphphos_model(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_heads=4,
                dropout_rate=dropout_rate
            )
            gnn_output = gnn_model([input_layer, adjacency_input])
        else:
            # 不使用GAT，就直接等于cnn_output
            gnn_output = cnn_output

        # 合并CNN和GNN的输出，如果都开，则concat；如果只开了一个，直接用那个
        if use_cnn and use_gat:
            merged_output = tf.concat([cnn_output, gnn_output], axis=-1)
        else:
            merged_output = gnn_output if use_gat else cnn_output

        # 如果使用LSTM
        if use_lstm:
            bilstm_output = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    64,
                    return_sequences=True,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    recurrent_dropout=0,
                    dropout=0.3
                )
            )(merged_output, mask=mask_input)
        else:
            # 不使用LSTM，就直接把merged_output传下去
            bilstm_output = merged_output

        # 如果使用Attention
        if use_attention:
            # 计算正确形状的attention_mask
            attention_mask = tf.logical_and(
                tf.expand_dims(mask_input, axis=1),
                tf.expand_dims(mask_input, axis=2)
            )
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=64
            )(bilstm_output, bilstm_output, attention_mask=attention_mask)
        else:
            attention_output = bilstm_output

        # 全连接层进行分类
        final_model = MultiFNN(hidden_units=hidden_channels, dropout_rate=dropout_rate)
        final_output = final_model(attention_output)

        # 构建模型
        model = tf.keras.Model(inputs=[input_layer, adjacency_input, mask_input], outputs=final_output)

        # 选择优化器
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

        # 编译模型，使用加权的二元交叉熵
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, class_weights),
            metrics=[f1_m, positive_accuracy_m, negative_accuracy_m, accuracy_m, precision_m, recall_m]
        )
    return model