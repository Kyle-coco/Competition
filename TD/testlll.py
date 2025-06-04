import pandas as pd

# 读取学习统计表
df = pd.read_csv('game_TD/user_learn_statistics_total.csv')

# 读取学习日志表
df2 = pd.read_csv('game_TD/activity_learn_log.csv')

# 合并两个表格，使用用户ID（userId）作为合并的键
merged_df = pd.merge(df, df2, on='userId')

#
merged_df.dropna(inplace=True)

# 输出
print(merged_df.head())

# output_e_path = 'ttt.xlsx'
# merged_df.to_excel(output_e_path, index=False)

# # 输出多列
# print(data[['姓名', '成绩']])
