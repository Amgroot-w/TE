"""
定义了三个类: ConvLayer 卷积层
            MaxPoolingLayer 池化层
            FullConnectedLayer 全连接层
"""
import numpy as np

# from activators import ReluActivator, IdentityActivator
# 定义激活函数及其导数
class ReluActivator(object):
    def forward(self, weighted_input):
        # return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0

class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output


# 获取卷积区域
def get_patch(input_array, i, j, filter_width,
              filter_height, stride):
    """
    参数: 输入矩阵
         输出矩阵的index（i,j）
         filter的宽、高
         步长
    """
    """
    从输入数组中获取本次卷积的区域，
    自动适配输入为2D和3D的情况
    """
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]


# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


# 计算卷积
def conv(input_array,
         kernel_array,
         output_array,
         stride, bias):
    """
    :param input_array: 输入特征体
    :param kernel_array: 滤波器filter（要学习的参数）
    :param output_array: 输出特征体（已经提前初始化好了，是一个全0的矩阵）
    :param stride: 步长
    :param bias: 偏重（要学习的参数）
    :return: 无返回值，此函数目的是计算output_array（直接改变通过形参传递进来的）
    """
    """
    ConvLayer内部函数
    计算卷积，自动适配输入为2D和3D的情况
    """

    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                                         get_patch(input_array, i, j, kernel_width,
                                                   kernel_height, stride) * kernel_array
                                 ).sum() + bias


# 为数组增加Zero padding
def padding(input_array, zp):
    """
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    """
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_depth = input_array.shape[0]
            input_height = input_array.shape[1]
            input_width = input_array.shape[2]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
            zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_height = input_array.shape[0]
            input_width = input_array.shape[1]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array


# 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        i[...] = op(i)


class Filter(object):
    """
    Filter即为CNN模型需要训练的参数（权重+偏置）
    """
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4,
                                         (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(
            self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        # 重写__repr__方法：为Filter类制作自定义的自我描述信息
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride,
                 activator=ReluActivator(), learning_rate=0.01):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride

        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding,
                stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding,
                stride)
        self.output_array = np.zeros((self.filter_number,
                                      self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter_height, self.channel_number))

        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        """
        计算卷积层的输出
        输出结果保存在self.output_array
        """
        self.input_array = input_array
        self.padded_input_array = padding(input_array,
                                          self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array,
                        self.activator.forward)

    def backward(self, input_array, sensitivity_array,
                 activator=ReluActivator()):
        """
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        """
        按照梯度下降，更新权重
        """
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        """
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        """
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)  # **解决步长为S的问题
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        """
        下面的公式说明：L-1层的delta_array，等于L层的expanded_array和filter的
        卷积（此卷积是数学上定义的卷积，即filter要翻转一下），而L-1层的delta_array
        的shape即为Class类的输入的shape，依此即可反向推出zp
        """
        zp = int((self.input_width + self.filter_width - 1 - expanded_width) / 2)
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            """
            np.rot90(x, k)：矩阵旋转90°（或90°的k倍）
            map(function, iterable)：根据提供的函数对指定序列做映射
            """
            flipped_weights = np.array(list(map(lambda i: np.rot90(i, 2), filter.get_weights())))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):  # **解决输入深度为D的问题
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array  # **解决filter个数为N的问题
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array):
        """
        当步长不等于1时，需要对sensitivity map进行扩展
        sensitivity_array的shape和self的输出特征体的shape一致
        """
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小，即stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,
                         self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size,
                              filter_size, zero_padding, stride):
        return int((input_size - filter_size +
                2 * zero_padding) / stride + 1)


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height, self.output_width))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                                     i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator=SigmoidActivator(), learning_rate=0.01):
        """
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.learning_rate = learning_rate
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
                                   (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        前向计算
        input_array: 输入向量，维度必须等于input_size
        """
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        """
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)

        self.b_grad = delta_array

    def update(self):
        """
        使用梯度下降算法更新权重
        """
        self.W -= self.learning_rate * self.W_grad
        self.b -= self.learning_rate * self.b_grad


# 以下为测试代码
def init_test():
    # a为输入
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    # b为sensitivity map
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl


def test():
    a, b, cl = init_test()
    cl.forward(a)
    print(cl.output_array)


def test_bp():
    a, b, cl = init_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])


def gradient_check():
    """
    梯度检查
    """
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape,
                                dtype=np.float64)
    # 计算梯度
    cl.backward(a, sensitivity_array, IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon  # 在加一个的基础上减两个，相当于减一个
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon  # 再加一个正好和前面抵消，保证检查完毕之后权重的值不变
                print('weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j]))


def init_pool_test():
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],
         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    b = np.array(
        [[[1, 2],
          [2, 4]],
         [[3, 5],
          [8, 2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)

    return a, b, mpl


def test_pool():
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print('input array:\n%s\noutput array:\n%s' % (a,
                                             mpl.output_array))


def test_pool_bp():
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (
        a, b, mpl.delta_array))


if __name__ == '__main__':
    print('*********************************')
    gradient_check()
    print('*********************************')
    test()
    print('*********************************')
    test_bp()
    print('*********************************')
    test_pool()
    print('*********************************')
    test_pool_bp()










