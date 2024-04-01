import numpy as np
import pandas as pd

# 读取CSV文件中的网球比赛数据
data = pd.read_csv('resource_data.csv')

#将elapsed_time转化为分钟单位
# 定义一个函数来将时间格式转换为分钟数
def convert_time_to_minutes(time_str):
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 60 + m + s / 60  # 将秒数转换为分钟
    except ValueError:
        return 0  # 对于无效格式，返回默认值或进行适当处理
# 将 elapsed_time 列应用上述函数进行转换
data['elapsed_time'] = data['elapsed_time'].apply(convert_time_to_minutes)

# 将 'p1_score' 列的AD映射为4，其他的分别映射
data['p1_score'] = data['p1_score'].replace({
    '15':1 , '30':2 , '40':3 , 'AD': 4
})
data['p1_score'] = pd.to_numeric(data['p1_score'], errors='coerce')
# 将 'p2_score' 列的AD映射为4，其他的分别映射
data['p2_score'] = data['p2_score'].replace({
    '15':1 , '30':2 , '40':3 , 'AD': 4
})
data['p2_score'] = pd.to_numeric(data['p2_score'], errors='coerce')
#每小局领先的分
data['game_score_lead'] = data['p1_score'] - data['p2_score']

# 将 'server' 列的2映射为-1，即-1代表player2
data['server'] = data['server'].replace({
    2: -1
})

# 将 'serve_no' 更新为 'server' 列和 'serve_no'列的逐元素相乘
data['serve_no'] = data['server'] * data['serve_no']

# 将 'point_victor' 列的2映射为-1，即-1代表player2
data['point_victor'] = data['point_victor'].replace({
    2: -1
})

# 添加新列 'set_score_lead' 到 data 数据框
data['set_score_lead'] = data['p1_points_won'] - data['p2_points_won']

# 将 'game_victor' 列的2映射为-1，即-1代表player2
data['game_victor'] = data['game_victor'].replace({
    2: -1
})

# 将 'set_victor' 列的2映射为-1，即-1代表player2
data['set_victor'] = data['set_victor'].replace({
    2: -1
})

# 定义需要累加的列
columns_to_accumulate = ['p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_double_fault', 'p2_double_fault',
                          'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won',
                          'p1_break_pt', 'p2_break_pt','p1_break_pt_won','p2_break_pt_won','p1_break_pt_missed','p2_break_pt_missed']

# 遍历每一列
for column in columns_to_accumulate:
    # 初始化变量用于记录上一个 set_no 的值
    prev_set_no = None
    # 遍历当前列和 'set_no' 列
    for index, (value, set_no_value) in enumerate(zip(data[column], data['set_no'])):
        # 如果 set_no 变化，将当前列值清零
        if prev_set_no is not None and set_no_value != prev_set_no:
            data.at[index, column] = 0
        else:
            # 否则，累加当前列值
            data.at[index, column] = data.at[index - 1, column] + value if index > 0 else value
        # 更新上一个 set_no 的值
        prev_set_no = set_no_value

# 添加新列，代表p1比p2多的数目
data['ace'] = data['p1_ace'] - data['p2_ace']
data['winner'] = data['p1_winner'] - data['p2_winner']
data['double_fault'] = data['p1_double_fault'] - data['p2_double_fault']
data['unf_err'] = data['p1_unf_err'] - data['p2_unf_err']
data['net_pt'] = data['p1_net_pt'] - data['p2_net_pt']
data['net_pt_won'] = data['p1_net_pt_won'] - data['p2_net_pt_won']
data['break_pt'] = data['p1_break_pt'] - data['p2_break_pt']
data['break_pt_won'] = data['p1_break_pt_won'] - data['p2_break_pt_won']
data['break_pt_missed'] = data['p1_break_pt_missed'] - data['p2_break_pt_missed']
data['distance_run'] = data['p1_distance_run'] - data['p2_distance_run']

# 将 'serve_width' 列的文本值映射为数字
data['serve_width'] = data['serve_width'].replace({
    'B': 1, 'BC': 2, 'BW': 2, 'C': 1, 'W': 1
})
# 将 'serve_width' 列乘以server更新
data['serve_width'] = data['serve_width']* data['server']

# 将 'serve_depth' 列的文本值映射为数字
data['serve_depth'] = data['serve_depth'].replace({
    'CTL': 1, 'NCTL': -1
})
# 将 'serve_depth' 列乘以server更新
data['serve_depth'] = data['serve_depth']* data['server']

# 将 'return_depth' 列的文本值映射为数字
data['return_depth'] = data['return_depth'].replace({
    'D': 1, 'ND': -1
})
# 将 'return_depth' 列乘以server更新
data['return_depth'] = data['return_depth']* data['server']

# 初始化变量用于记录连胜信息
p1_count = 0
p2_count = 0
# 遍历 'point_victor' 列
for index in range(0, len(data)):
    # 检查是否连胜
    if index > 0 :
        if data.at[index -1, 'point_victor'] == 1:
            p1_count += 1
            p2_count = 0
        if data.at[index -1, 'point_victor'] == -1:
            p1_count = 0 
            p2_count += 1
    data.at[index, 'p1_streak'] = p1_count
    data.at[index, 'p2_streak'] = p2_count

# 添加新列 'streak' 
data['streak'] = data['p1_streak'] - data['p2_streak']

# 添加新列 score_sum
data['score_sum_p1']=data["p1_points_won"].diff(5).fillna(0)
data['score_sum_p2']=data["p2_points_won"].diff(5).fillna(0)
# 删除包含缺失值的行（有一些行的速度是NAN）
data = data.dropna(subset=['speed_mph'])
data = data.dropna(subset=['return_depth'])

# 将处理后的数据保存为 CSV 文件
data.to_csv('processed_data.csv', index=False)