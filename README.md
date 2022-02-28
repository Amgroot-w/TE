# 毕设TE

小论文：《基于LSTM-DAE的化工故障诊断方法研究》：https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=BJHY202102014&uniplatform=NZKPT&v=5OxacFuB0g0t-8sc6diq6ccHqUGGhq0R5xACgSE3P_gfqqXkIaD1DA9ZSXHBkQD6

2021.6.10更新

+ **项目环境：python3.6**

+ **项目依赖包见文件：requirements.txt**

## 1.代码
### 1.1 模型代码
**版本1.0**：包含TE_main1（5个）和TE_main2（4个），前者用于特征提取，后者用于分类器。此版本模型均为二分类，且前者提取到的特征会保存为csv文件，后者直接读取csv文件训练二分类器，并做预测。
*第一版的一个未完成代码：TE_main_tSNE.py，该代码是为了复现 “SAE+t-SNE+DBSCAN&K-means”这篇论文，但是没有完成.*

**版本2.0**：包含TE_Main_DAE+Softmax.py和TE_Main_LSTM&DAE+Softmax.py两个文件（“预训练 + 微调”）。

**版本3.0**：包含TE_FinalModel1 ~ TE_FinalModel10共10个文件。这个版本的模型思想是：将正常样本和故障样本分开提取特征，分开训练二分类器（也是每个故障单独研究），效果极好，最后一个LSTM模型直接训练、测试全部达到了100%，这是因为人为干预了分类器的训练，所以这个版本模型不符合实际规律。

**版本4.0**：TE_FinalModel1 ~ TE_FinalModel10中，6个命名为~~.2的文件。这个版本模型重新使用了之前的方
法，先提取特征，然后训练分类器，只不过这次是在一个脚本中完成，不再像版本1.0一样分开，而且全部用类class实现了。但是4.0版本模型的整体效果比1.0版本好的原因并不是以上两点，而是4.0版本重新选择了故障类别，选了BP难以区分而自编码器好区分的故障类别，所以精度就上去了，但是LSTM没有调好，精度仍然低。

**版本5.0（最终版本）**：包含TE_FinalMultiModel1 ~ TE_FinalMultiModel6共6个文件。此版本是最终的多分类模
型，之前的多分类做不好是因为之前故障类别没有选好，这次认真研究了所有故障，选出了6类故障，这6类故障放在一块组成的数据集的特点是：BP和SVM直接分类的效果不好，但是经过特征提取后效果很好。最终实验结果表明分类效果正好是递进的关系，至此，我们完美完成了TE的任务！（只能说完成了部分TE故障的分类任务，因为对于有些故障，最终版本的模型也无能为力）

### 1.2 TE子函数代码
**TE_function.py**: 集成了TE项目所有的子函数，实现了项目必须的12个重要功能（包括5个模型类，1个数据集类，6个功能子函数）；

**TE_data_str2float.py**: TE原始数据的第一步处理：转化字符型数据，顺便填补了缺失值；

**TE_MI**: 计算52个变量之间的互信息MI；

**TE_plot.py**: 实现画图功能，集成了好几个画图函数；

**TE_GridSearch_LSTM.py**: 网格搜索寻优。

### 1.3 其他代码
**TE_OneClass_、TE_MultiClass_、CNNlayers、Cancer_**：学习pipeline、RNN、CNN时的代码；

**AAAAACodeForTest.py**： 每个项目必有的工具代码；


## 2.数据
**TE_data**: TE过程原始数据集；

**TE_Train_data.csv**: 第一步处理后的TE训练集；

**TE_Test_data.csv**: 第一步处理后的TE测试集；

**train_feature.csv**: 版本1.0模型保存的训练集特征数据；

**test_feature.csv**: 版本1.0模型保存的测试集特征数据；

**accuracy_train.csv、accuracy_test.csv**：准确率数据；

**MNIST_data**: 手写数字识别数据集;

**plot_ac.csv**: 写论文的时候用到的画图数据。

## 3.TensorFlow保存的ckpt模型
**SaveTE、SaveTE_Feature、TE_SaveDAE、TE_SaveLSTM、TE_SaveMainModel、TE_SaveMulti**: 各种版本的模型。

## 4.图片
**images1.0**：3.0版本模型保存的图片；

**images2.0**：4.0版本模型保存的图片；

**images3.0**：5.0版本模型保存的图片，保存了论文中用到的图片。

**plot3.png**: TE_plot画的图。

## 5.其他



















