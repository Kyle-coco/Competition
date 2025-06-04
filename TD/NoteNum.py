import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取 CSV 文件
classroom_member_path = "game_TD/classroom_member.csv"  # 替换为实际的文件路径
classroom_member_df = pd.read_csv(classroom_member_path)


print(classroom_member_df.head())

# 按班级 ID 划分，计算每个班级的笔记数和话题数的总和
classroom_summary = classroom_member_df.groupby('classroomId').agg({
    'noteNum': 'sum',
    'threadNum': 'sum'
}).reset_index()

# 可视化柱状图
plt.figure(figsize=(10, 6))
plt.bar(classroom_summary['classroomId'], classroom_summary['noteNum'], label='笔记总数', color='skyblue')
plt.bar(classroom_summary['classroomId'], classroom_summary['threadNum'], bottom=classroom_summary['noteNum'], label='话题总数', color='orange')
plt.xlabel('班级ID')
plt.ylabel('数量')
plt.title('每个班级的笔记数和话题数总和')
plt.legend()
plt.grid(True)
plt.show()
