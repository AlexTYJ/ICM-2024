# 导入所需的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件中的网球比赛数据
data = pd.read_csv('processed_data.csv')

# 选择特征和目标变量
features = data[['set_score_lead','game_score_lead','streak',
                 'server','serve_no','serve_width','serve_depth',
                 'ace','winner','double_fault','unf_err','net_pt','net_pt_won','break_pt','break_pt_won',
                 'distance_run','speed_mph','rally_count','return_depth'
                 ]]
target = data['point_victor']

# 将所有特征转换为数值型，并处理非数值值
features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

# 对特征进行标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#将数据集划分为训练集和测试集，随机划分
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=10)

# 非随机划分
#split_index = int(0.7 * len(features_scaled))
#X_train, X_test = features_scaled[:split_index], features_scaled[split_index:]
#y_train, y_test = target[:split_index], target[split_index:]


# 初始化逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测并评估模型准确性
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# 在测试集上进行预测
y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取预测概率，通常选择第二列即正类别的概率

# 将 y_test 和 y_pred 转换为 DataFrame
result_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred,'y_pred_prob':y_pred_prob})

# 导出到 CSV 文件
result_df.to_csv('predictions.csv', index=False)

# 获取系数值和对应的因素名称
coefficients = model.coef_[0]
feature_names = features.columns

# 创建 DataFrame 以便使用 Seaborn 绘图
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# 根据系数值大小排序
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# 使用 Seaborn 绘制条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.show()

# 将系数数据保存到 CSV 文件
coef_df.to_csv('logistic_regression_coefficients.csv', index=False)
