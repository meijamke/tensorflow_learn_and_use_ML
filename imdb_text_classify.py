"""
    日期：2019.4.21
    作者：meijamke
    功能：将文本形式的影评分为“正面”或“负面”——二元分类

    隐藏单元（神经单元）：


    问题：由于tensorflow版本过低，tf.keras.dataset里并没有Fashion MNIST 数据集文件夹
    解决：使用keras库的数据集
        用法：
        from keras.datasets import imdb
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
"""
# tensorflow, keras and datasset
import tensorflow as tf
import keras
from keras.datasets import imdb

# lib helper
import numpy as np

# 显示tf版本
# print(tf.__version__)
"""
    加载数据
    为确保数据规模处于可管理的水平，参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。
    数据集已经进行了预处理，影评（字词序列）被转换为整数序列，其中每个整数表示字典中的一个特定字词。
"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""
    探索数据
    数据集已经过预处理：每个样本都是一个整数数组，表示影评中的字词。
    每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。
"""
# 训练集数据和标签数量
print('train_entries:{}, labels:{}'.format(len(train_data), len(train_labels)))

# 训练集第一个样本
print(train_data[0])

# 影评的长度可能会有所不同。由于神经网络的输入必须具有相同长度，所以这需要在后续处理（填充）
print(len(train_data[0]), len(train_data[1]))

# 将整数转换成字词，查看字词
# reverse_word_index = {'字词': 整数}
word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = {v: k for k, v in word_index.items()}


def decode_review(text):
    return ' '.join(reverse_word_index.get(i, '?') for i in text)


# 使用decode_review查看第一个样本的字词
print(decode_review(train_data[0]))

"""
    预处理数据
    将影评（整数数组）转换为张量，将张量馈入网络。整数数组转换为张量有两种方法：
        1.one-hot编码，使每一个样本的字词编码成1和0的向量，构成一个num_words（10000）*num_reviews（10000）的张量
        2.padding数据，使每一个样本（整数数组）的长度一致，构成一个max_length（256）*num_reviews（10000）的张量1
"""

train_data = keras.preprocessing.sequence.pad_sequences(
                                    train_data,
                                    padding='post',
                                    maxlen=256,
                                    value=word_index['PAD']
                                )

test_data = keras.preprocessing.sequence.pad_sequences(
                                    test_data,
                                    padding='post',
                                    maxlen=256,
                                    value=word_index['PAD']
                                )

# 查看填充数据后数据的长度
print(len(train_data[0]), len(train_data[1]))

# 查看第一条影评（整数数组）
print(train_data[0])

"""
    构建模型：
    配置模型+编译模型
"""

"""
    配置模型
    神经网络的基本构造块是层。层从馈送到其中的数据中提取表示结果，
    并且希望这些表示结果有助于解决手头问题。
    大部分深度学习都会把简单的层连在一起。
    大部分层（例如 tf.keras.layers.Dense）都具有在训练期间要学习的参数。
    
    神经网络通过堆叠层创建而成，这需要做出两个架构方面的主要决策：     
        1.要在模型中使用多少个层？
        2.要针对每个层使用多少个隐藏单元？
    
    第一个层是映射层，包含10000神经单元，该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。模型在接受训练时会学习这些向量。
                    这些向量会向输出数组添加一个维度。生成的维度为：(batch, sequence, embedding)
    第二个层是全局池化层，减少参数量。通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。
                    这样，模型便能够以尽可能简单的方式处理各种长度的输入。
    第三个层是密集连接（全连接层），包含16个神经单元，relu激活函数，将<0的置为0，>0的保持不变。
    第四个层是密集连接（全连接层），包含1个神经单元，sigmoid激活函数，将数值映射到(0,1)之间，作为输出的预测概率。
"""
vocab_size = 10000

# 第一种配置模型的方法
# model = keras.Sequential([
#     keras.layers.Embedding(vocab_size, 16),
#     keras.layers.GlobalAveragePooling1D(),
#     keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
#
# model.summary()

# 第二种配置模型的方法
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D)
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

"""
    隐藏单元（神经单元）：
        上述模型在输入和输出之间有两个中间层（也称为“隐藏”层）。
        输出（单元、节点或神经元）的数量是相应层的表示法空间的维度。换句话说，该数值表示学习内部表示法时网络所允许的自由度。
    
        如果模型具有更多隐藏单元（更高维度的表示空间）和/或更多层，则说明网络可以学习更复杂的表示法。
        不过，这会使网络耗费更多计算资源，并且可能导致学习不必要的模式（可以优化在训练数据上的表现，但不会优化在测试数据上的表现）。这称为过拟合。
"""

"""
    编译模型
    由于这是一个二元分类问题且模型会输出一个概率（应用 S 型激活函数的单个单元层），因此我们将使用 binary_crossentropy 损失函数。

    该函数并不是唯一的损失函数，例如，您可以选择 mean_squared_error。
    但一般来说，binary_crossentropy 更适合处理概率问题，它可测量概率分布之间的“差距”，在本例中则为实际分布和预测之间的“差距”。
"""
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.train.AdamOptimizer(),
    metrics=['accuracy']
)



