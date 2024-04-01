import pandas as pd

# 从CSV文件中读取数据
data = pd.read_csv('processed_data.csv')
scores = pd.read_csv('scores_p2.csv')

# 提取 "elapsed_time" 列和 "scores" 列
t = data["elapsed_time"]
scores = scores.iloc[:len(t)]  # 确保 scores 和 t 的长度一致

# 对 "elapsed_time" 列和 "scores" 列进行差分
diff_t = t.diff().dropna().values.flatten()
diff_scores = scores.diff().dropna().values.flatten()

# 计算差分比值
m= diff_scores / diff_t[:len(diff_scores)]
momentum = pd.DataFrame({'time': t.iloc[0:len(m)], 'momentum':m , 'capacity': data["score_sum_p2"].iloc[0:len(m)]})

# 将结果保存到CSV文件
momentum.to_csv('momentum_p2.csv', index=False)
