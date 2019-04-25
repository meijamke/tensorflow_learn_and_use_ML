"""
    日期：2019.4.19
    作者：meijamke
    功能：训练首个神经网络：对10种类别服饰图像分类
    Fashion MNIST 数据集：
        其中包含 70000 张灰度图像，涵盖 10 个类别
        图像为 28x28 的 NumPy 数组，像素值介于 0 到 255 之间。
        标签是整数数组，介于 0 到 9 之间。
    Fashion MNIST 数据集相对较小，适合用于验证某个算法能否如期正常运行。
    是测试和调试代码的良好起点。

    本神经网络使用 60000 张图像训练网络，
    并使用 10000 张图像评估经过学习的网络分类图像的准确率。

    问题：由于tensorflow版本过低，tf.keras.dataset里并没有Fashion MNIST 数据集文件夹
    解决：使用keras库的数据集
        用法：
        from keras.datasets import fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
"""
# TensorFlow and keras and dataset
import tensorflow as tf
import keras
from keras.datasets import fashion_mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 解决plt.show()中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 显示tf版本
# print(tf.__version__)
"""
    导入 Fashion MNIST 数据集
"""

# fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
    每张图像都映射到一个标签（0~9）
"""
# 其对应的服饰类别如下
# 便于后面绘制图形表时使用
class_name = ['T 恤衫/上衣', '裤子', '套衫', '裙子', '外套', '凉鞋', '衬衫', '运动鞋', '包包', '踝靴']

"""
    探索训练数据集的格式
"""
# 可以看到训练集有60000张图像
# 每张图像都表示为28x28像素
# 每个标签都是一个介于0~9的整数
print(train_images.shape)
print(len(train_labels))
print(train_labels)

"""
    探索测试数据集的格式
"""
# 可以看到测试集有10000张图像
# 每张图像都表示为28x28像素
# 每个标签都是一个介于0~9的整数
print(test_images.shape)
print(len(test_labels))
print(test_labels)

# 检查训练集中的第一张图像
plt.figure(figsize=(5, 5))
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

"""
    预处理数据：将0~255像素值缩放到0~1之间
                然后将其馈送到神经网络模型
"""
train_images = train_images/255
test_images = test_images/255

# 绘制训练集前25张图像，并在每张图像下显示类别名称
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_name[train_labels[i]])
plt.show()


"""
    构建模型：
        1.配置模型
        2.编译模型
"""
"""
    1.配置模型：
    神经网络的基本构造块是层。层从馈送到其中的数据中提取表示结果，
    并且希望这些表示结果有助于解决手头问题。
    大部分深度学习都会把简单的层连在一起。
    大部分层（例如 tf.keras.layers.Dense）都具有在训练期间要学习的参数。
        
    神经网络通过堆叠层创建而成，这需要做出两个架构方面的主要决策：     
        1.要在模型中使用多少个层？
        2.要针对每个层使用多少个隐藏单元？
    
    该网络包含3个层，
    第一个层将多维数据压扁成一维，含有784个神经单元
    第二个层是密集连接（全连接层），含有128个神经单元，relu激活函数，将<0的置为0，>0的保持不变。
    第三个层是密集连接（全连接层），含有10个神经单元，softmax激活函数，将数值映射到(0,1)之间，作为类别预测概率。
    
    第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素）。
    可以将该层视为图像中像素未堆叠的行，并排列这些行。该层没有要学习的参数；它只改动数据的格式。
    在扁平化像素之后，该网络包含两个 tf.keras.layers.Dense 层的序列。这些层是密集连接或全连接神经层。
    第一个 Dense 层具有 128 个节点（或神经元）。第二个（也是最后一个）层是具有 10 个节点的 softmax 层，该层会返回一个具有 10 个概率得分的数组，这些得分的总和为 1。
    每个节点包含一个得分，表示当前图像属于 10 个类别中某一个的概率。
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
    2.编译模型
    模型还需要再进行几项设置才可以开始训练。这些设置会添加到模型的编译步骤：
    
    损失函数 - 衡量模型在训练期间的准确率。我们希望尽可能缩小该函数，以“引导”模型朝着正确的方向优化。
    优化器 - 根据模型看到的数据及其损失函数更新模型的方式。
    指标 - 用于监控训练和测试步骤。以下示例使用准确率，即图像被正确分类的比例。
"""
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.train.AdamOptimizer(),
    metrics=['accuracy']
)
"""
    训练模型
    训练神经网络模型需要执行以下步骤：
        
        1.将训练数据馈送到模型中，在本示例中为 train_images 和 train_labels 数组。
        2.模型学习将图像与标签相关联。
        3.我们要求模型对测试集进行预测，在本示例中为 test_images 数组。我们会验证预测结果是否与 test_labels 数组中的标签一致。
        
    要开始训练，请调用 model.fit 方法，使模型与训练数据“拟合”：
"""
model.fit(train_images, train_labels, epochs=4)

"""
    评估准确率
    如果机器学习模型在新数据上的表现不如在训练数据上的表现，就表示出现过拟合。
"""
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_loss, test_acc)

"""
    做出预测
"""
# 对每个测试服饰的类别做预测，返回预测得到的样本数*10个置信度
predictions = model.predict(test_images)
print(predictions[0])

# np.argmax()，返回指定维度最大值的索引号
# 这里是得到对第一个样本预测结果中置信度最高的服饰类别
pred = np.argmax(predictions[0])
print(pred)

# 实际的服饰类别
print(test_labels[0])

"""
    将预测结果绘制成图来查看全部 10 个类别的得分
"""


# 绘制测试图像
def plot_image(n, pred_array, true_label, img):
    pred_array, true_label, img = pred_array[n], true_label[n], img[n]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    pred_label = np.argmax(pred_array)
    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(
        class_name[int(pred_label)],
        100*np.max(pred_array),
        class_name[true_label]
    ), color=color)


# 绘制预测结果
def plot_value_array(n, pred_array, true_label):
    pred_array, true_label = pred_array[n], true_label[n]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), pred_array, color='#777777')
    plt.ylim([0, 1])
    pred_label = np.argmax(pred_array)

    thisplot[pred_label].set_color('red')
    thisplot[true_label].set_color('green')


"""
    绘制前25个测试集的图片、预测类别、真实类别
    x_label和柱形图颜色为绿色代表正确
"""
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

"""
    使用经过训练的模型对单个图像进行预测
    
    由于tf.keras 模型已经过优化，可以一次性对样本批次或样本集进行预测。
    因此，即使我们使用单个图像，仍需要将其添加到列表中
"""
# 获取单个图像
img = test_images[0]
print(img.shape)

# 扩展图像维度
img = np.expand_dims(img, 0)
print(img.shape)

# 预测单个图像
predictions_single = model.predict(img)
print(predictions_single)

# 绘制预测结果的条形图
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_name, rotation=45)
plt.show()

# 输出预测结果中置信度最高的类别
pred_single = np.argmax(predictions_single)
print(pred_single)
