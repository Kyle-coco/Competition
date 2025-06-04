import pandas as pd
import matplotlib.pyplot as plt

# 读取学习统计表
user_learn_df = pd.read_csv('game_TD/user_learn_statistics_total.csv')

# 读取学习日志表
activity_log_df = pd.read_csv('game_TD/activity_learn_log.csv')

# 合并两个表格，使用用户ID（userId）作为合并的键
merged_df = pd.merge(user_learn_df, activity_log_df, on='userId')

# 去除缺失值
merged_df.dropna(inplace=True)

# 对学习时长进行分析，例如计算每个用户的总学习时长
total_learned_time = merged_df.groupby('userId')['learnedTime'].sum()

# 导出文件csv的格式
# output_csv_path = 'studying_time.csv'
# total_learned_time.to_csv(output_csv_path, index=False)

# 输出每个用户的总学习时长
print(total_learned_time)

# 设置图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决符号显示问题

# 生成散点图，调整点大小和透明度
plt.figure(figsize=(10, 6))
plt.scatter(total_learned_time.index, total_learned_time.values, s=50, alpha=0.7)
plt.xlabel('用户ID')
plt.ylabel('总学习时长')
plt.title('每个用户的总学习时长')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
