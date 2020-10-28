import pandas as pd
df = pd.read_csv('D:\Python Codes\TE\Cancer_wdbc.csv', header=None)  # 读取数据集

from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values # 抽取训练集特征
y = df.loc[:, 1].values # 抽取训练集标签
le = LabelEncoder()
y = le.fit_transform(y) # 把字符串标签转换为整数，恶性-1，良性-0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1) # 拆分成训练集(80%)和测试集(20%)

# 下面，我要用逻辑回归拟合模型，并用标准化和PCA（30维->2维）对数据预处理，用Pipeline类把这些过程链接在一起
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 用StandardScaler和PCA作为转换器，LogisticRegression作为评估器
estimators = [('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))]
# Pipeline类接收一个包含元组的列表作为参数，每个元组的第一个值为任意的字符串标识符，比如：我们可以通过pipe_lr.named_steps['pca']来访问PCA组件;第二个值为scikit-learn的转换器或评估器
pipe_lr = Pipeline(estimators)
pipe_lr.fit(X_train, y_train)
print(pipe_lr.score(X_test, y_test))
