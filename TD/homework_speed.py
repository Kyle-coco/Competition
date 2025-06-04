import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("D:/PyCharm/TD_data_analysis/game_TD/user_learn_statistics_total.csv")

# 删除含有缺失值的行
data = data.dropna(subset=['learnedSeconds', 'finishedTaskNum'])

# 去除学习时长为0的记录
data = data[data['learnedSeconds'] != 0]

# 计算学习小时数和学习速率
data['learnedHours'] = data['learnedSeconds'] / 3600  # 秒转小时
data['速率'] = data['finishedTaskNum'] / data['learnedSeconds']

# 对学生速率进行排序（降序）
sorted_data = data.sort_values(by='速率', ascending=False)

# 如果学生数量超过20名，则随机抽样20名学生
if len(sorted_data) > 20:
    sampled_data = sorted_data.sample(n=20, random_state=42)
else:
    sampled_data = sorted_data

# 设置中文字体和图形参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams["font.size"] = 10               # 字体大小
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 可视化学生速率
plt.figure(figsize=(10, 6))
plt.bar(sampled_data['userId'].astype(str), sampled_data['速率'])  # 转为字符串以避免整数问题
plt.xlabel('学生ID')
plt.ylabel('速率')
plt.title('学生完成课程材料的速率（随机抽样20人）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 导出分析结果
sampled_data.to_excel('analyzed_student_data_sampled.xlsx', index=False)
sampled_data.to_csv('analyzed_student_data_sampled.csv', index=False)

# 显示最终速率数据
result = sampled_data[['userId', '速率']]
print(result)
