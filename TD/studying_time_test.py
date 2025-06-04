# 最终结果

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 数据读取
user_learn_df = pd.read_csv('game_TD/activity_learn_log.csv')
activity_log_df = pd.read_csv('game_TD/student_course_participation_counts.csv')

# 数据合并
merged_df = pd.merge(user_learn_df, activity_log_df, on='userId', how='inner')

# 清洗数据：仅移除特定列的缺失值
merged_df.dropna(subset=['learnedTime'], inplace=True)

# 将时间戳转换为小时
merged_df['learnedTime'] = merged_df['learnedTime'] / 3600  # 假设learnedTime是以秒为单位的时间戳

# 数据分析
total_learned_time = merged_df.groupby('userId')['learnedTime'].sum()
time_bins = [x for x in range(0, 55, 1)]  # 从0到50，步长为5
  # 定义学习时长区间（以小时为单位）
user_counts = pd.cut(total_learned_time, bins=time_bins).value_counts().sort_index()

# 图表设置
plt.figure(figsize=(12, 8))
ax = user_counts.plot(kind='bar', color='teal')
plt.xlabel('总学习时长区间 (小时)')
plt.ylabel('用户数量')
plt.title('不同学习时长区间的学生用户数量分布')
plt.xticks(rotation=45)
plt.grid(True)

# 在柱状图上方显示具体数值
for i in ax.patches:
    ax.text(i.get_x() + i.get_width()/2, i.get_height() + 0.5, str(i.get_height()), ha='center', va='bottom')

plt.tight_layout()
plt.show()
