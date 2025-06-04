import pandas as pd

# 读取 CSV 文件
user_stats_path = 'game_TD/user_learn_statistics_total.csv'
classroom_member_path = 'game_TD/classroom_member.csv'

user_stats_df = pd.read_csv(user_stats_path)
classroom_member_df = pd.read_csv(classroom_member_path)

# 连接用户学习统计表和班级成员表
merged_df = pd.merge(user_stats_df, classroom_member_df, on='userId')

# 确保学习时长转换为小时，避免秒数带来的大数问题
merged_df['learnedHours'] = merged_df['learnedSeconds'] / 3600

# 计算每个班级的完成任务的速率（每小时完成的任务数）
# grouped_df 会包含每个班级的任务完成速率
grouped_df = merged_df.groupby('classroomId').apply(
    lambda x: x['finishedTaskNum'].sum() / x['learnedHours'].sum() if x['learnedHours'].sum() > 0 else 0
).reset_index(name='taskCompletionRatePerHour')


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(grouped_df['classroomId'], grouped_df['taskCompletionRatePerHour'], color='blue')
plt.title('按班级计算每小时任务完成率')
plt.xlabel('Classroom ID')
plt.ylabel('每小时任务完成率')
plt.grid(True)
plt.show()
