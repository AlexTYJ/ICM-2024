import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_csv('processed_data.csv')

# 提取所需的列
#selected_columns = ['server','winner','set_score_lead','net_pt_won','double_fault','serve_depth','distance_run','rally_count','ace',
#                    'game_score_lead','streak','serve_no','serve_width','unf_err','net_pt','break_pt','break_pt_won','speed_mph','return_depth'
#                    ]

#player1所需要的列
#selected_columns = ['server','winner','p1_points_won','p1_net_pt_won','p1_double_fault','serve_depth','p1_distance_run','rally_count','p1_ace',
#                    'p1_score','p1_streak','serve_no','serve_width','p1_unf_err','p1_net_pt','p1_break_pt','p1_break_pt_won','speed_mph','return_depth'
#                    ]
#player2所需要的列
selected_columns = ['server','winner','p2_points_won','p2_net_pt_won','p2_double_fault','serve_depth','p2_distance_run','rally_count','p2_ace',
                    'p2_score','p2_streak','serve_no','serve_width','p2_unf_err','p2_net_pt','p2_break_pt','p2_break_pt_won','speed_mph','return_depth'
                    ]
data_selected = data[selected_columns]
# 处理NaN值
imputer = SimpleImputer(strategy='mean')  # 也可以选择其他策略，比如'median'或'most_frequent'
data_imputed = pd.DataFrame(imputer.fit_transform(data_selected), columns=selected_columns)

# 数据归一化
i=8
data_normalized=data_imputed
data_normalized.iloc[:, :i+1]=(0.998*(data_normalized.iloc[:, :i+1]-np.min(data_normalized.iloc[:, :i+1]))/(np.max(data_normalized.iloc[:, :i+1])-np.min(data_normalized.iloc[:, :i+1])))+0.002
data_normalized.iloc[:, i:]=(0.998*(np.max(data_normalized.iloc[:, i:])-data_normalized.iloc[:, i:])/(np.max(data_normalized.iloc[:, i:])-np.min(data_normalized.iloc[:, i:])))+0.002
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_normalized)

# 计算权重（熵权法）
matrix = np.array(data_normalized)
epsilon = 1e-10
matrix = matrix + epsilon  # 添加一个小的正数，避免零值或负值
p=matrix/np.nansum(matrix,axis=0)
entropy = (-1/np.log2(len(matrix))) * np.nansum(p * np.log2(p),axis=0)
weights = 1 - entropy
weights /= np.sum(weights)


# 使用PCA进行降维
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_normalized * weights)

# 计算TOPSIS
ideal_best = np.max(data_pca, axis=0)
ideal_worst = np.min(data_pca, axis=0)

distance_best = np.sqrt(np.sum((data_pca - ideal_best) ** 2, axis=1))
distance_worst = np.sqrt(np.sum((data_pca - ideal_worst) ** 2, axis=1))

topsis_score = distance_worst / (distance_best + distance_worst)

# 输出TOPSIS
data['topsis_score'] = topsis_score
weights_df = pd.DataFrame({ 'topsis_score': topsis_score})
weights_df.to_csv('topsis_score.csv', index=False)

# 将权重保存到CSV文件_player
weights_df = pd.DataFrame({'Criteria': selected_columns, 'Entropy_Weight': weights})
weights_df.to_csv('entropy_weights_p2.csv', index=False)

# 定义正相关和负相关的权重
positive_weights = weights[:i]
negative_weights = weights[i+1:]

# 将数据矩阵按正负相关权重分割并分别加权
positive_matrix = data_normalized[:, :i] * positive_weights
negative_matrix = data_normalized[:, i+1:] * negative_weights

# 计算每个样本的正相关和负相关打分
positive_scores = positive_matrix.sum(axis=1)
negative_scores = negative_matrix.sum(axis=1)

# 计算最终的 TOPSIS 分数（正相关打分减去负相关打分）
topsis_scores = positive_scores - negative_scores

# 将 TOPSIS 分数进行 Min-Max 归一化
min_score = topsis_scores.min()
max_score = topsis_scores.max()
normalized_topsis_scores = (topsis_scores - min_score) / (max_score - min_score)

np.savetxt("scores_p2.csv", normalized_topsis_scores )

#排序
weights_df_sorted = weights_df.sort_values(by='Entropy_Weight', ascending=False)

# 可视化各个部分权重
plt.bar(weights_df_sorted['Criteria'], weights_df_sorted['Entropy_Weight'])
plt.title('Entropy Weight of Criteria')
plt.xlabel('Criteria')
plt.ylabel('Weight')
plt.xticks(rotation=45, ha='right')
plt.show()

