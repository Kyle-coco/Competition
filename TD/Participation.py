import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 读取用户角色和行为日志数据
classroom_member_path = "D:/PyCharm/TD_data_analysis/game_TD/classroom_member.csv"
log_path = "D:/PyCharm/TD_data_analysis/game_TD/log.csv"

classroom_member_df = pd.read_csv(classroom_member_path)
log_df = pd.read_csv(log_path)

# 调整行为权重
action_weights = {
    'login_success': 0.25, 'join_classroom': 0.75, 'add_student': 0.5, 'download': 0.5,
    'create_course': 1.25, 'join_course': 1, 'publish_course': 1, 'update_course': 0.75,
    'create_lesson': 1, 'delete_course': -0.5, 'user_login': 0.25, 'user_logout': -0.25,
    'add_task': 0.75, 'delete_task': -0.5, 'create_thread': 0.75, 'update_member': 0.5,
    'delete_member': -0.5, 'register': 1.25, 'password-changed': 0.25
    # 可以根据需要继续调整其他行为的权重
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
sns.histplot(user_engagement_score['engagementScore'], bins=20, kde=True)
plt.title('学生参与度得分分布')
plt.xlabel('参与度得分')
plt.ylabel('频率')
plt.show()

