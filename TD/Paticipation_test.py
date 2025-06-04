import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取用户角色和行为日志数据
classroom_member_path = "D:/PyCharm/TD_data_analysis/game_TD/classroom_member.csv"
log_path = "D:/PyCharm/TD_data_analysis/game_TD/log.csv"

classroom_member_df = pd.read_csv(classroom_member_path)
log_df = pd.read_csv(log_path)

# 定义行为权重
action_weights = {
    'login_success': 1, 'join_classroom': 2, 'add_student': 1, 'download': 1,
    'create_course': 3, 'join_course': 2, 'publish_course': 2, 'update_course': 1,
    'create_lesson': 2, 'delete_course': -1, 'user_login': 1, 'user_logout': -1,
    'add_task': 2, 'delete_task': -1, 'create_thread': 2, 'update_member': 1,
    'delete_member': -1, 'register': 3, 'password-changed': 1
    # 可以继续添加其他行为和权重
}

# 筛选出学生用户ID
students_df = classroom_member_df[classroom_member_df['role'].str.contains('|student|')]
student_ids = students_df['userId'].unique()

# 过滤 log 表，仅包含学生的行为记录
log_df = log_df[log_df['userId'].isin(student_ids) & log_df['action'].isin(action_weights.keys())]

# 映射行为到权重，计算参与度得分
log_df['weight'] = log_df['action'].map(action_weights)

# 计算每个学生的总参与度得分
user_engagement_score = log_df.groupby('userId')['weight'].sum().reset_index(name='engagementScore')

# 导出结果到 CSV
output_csv_path = 'student_engagement_scores.csv'
user_engagement_score.to_csv(output_csv_path, index=False)

print(f"Student engagement scores have been successfully exported to {output_csv_path}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.figure(figsize=(10, 6))
sns.histplot(user_engagement_score['engagementScore'], bins=30, kde=True)
plt.title('学生参与度得分分布')
plt.xlabel('参与度得分')
plt.ylabel('频率')
plt.xlim(0, 20000)  # 设置 x 轴区间
plt.show()
