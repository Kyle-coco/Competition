"""
要统计学生整体参与学习平台程度的指标，我们可以利用 log 表中的 action 字段来创建一个“参与度指数”。这个指数将根据学生的活动（如登录、加入班级、下载等）的类型和频率来衡量。

为了实现这一点，我们可以为不同类型的行为赋予不同的权重，这些权重反映了各种行为对于学生参与度的贡献大小。例如，我们可以认为加入课程或班级的行为比更改头像更能反映学生的参与度。

以下是一种可能的方法来计算参与度指数，首先定义各种行为的权重，然后根据学生的行为频率和这些权重来计算每个学生的参与度指数：
定义行为权重：为 log 表中的行为定义权重。
计算参与度指数：对每个学生的行为使用权重进行加权，计算总分。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 读取用户角色和行为日志数据
classroom_member_path = "D:/PyCharm/TD_data_analysis/game_TD/classroom_member.csv"
log_path = "D:/PyCharm/TD_data_analysis/game_TD/log.csv"

classroom_member_df = pd.read_csv(classroom_member_path)
log_df = pd.read_csv(log_path)

# # 定义行为权重
# action_weights = {
#     'login_success': 1, 'join_classroom': 3, 'add_student': 2, 'download': 2,
#     'create_course': 5, 'join_course': 4, 'publish_course': 4, 'update_course': 3,
#     'create_lesson': 4, 'delete_course': -2, 'user_login': 1, 'user_logout': -1,
#     'add_task': 3, 'delete_task': -2, 'create_thread': 3, 'update_member': 2,
#     'delete_member': -2, 'register': 5, 'password-changed': 1
#     # 可以继续添加其他行为和权重
# }

# 调整行为权重（减小为原来的四分之一）
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

