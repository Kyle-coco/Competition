import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建示例数据
data = {
    'classroomId': [1, 2, 3, 4, 5],
    'noteNum': [10, 15, 8, 12, 20],
    'threadNum': [5, 7, 3, 8, 10]
}
classroom_member_df = pd.DataFrame(data)

# 按班级 ID 划分，计算每个班级的笔记数和话题数的总和
classroom_summary = classroom_member_df.groupby('classroomId').agg({
    'noteNum': 'sum',
    'threadNum': 'sum'
}).reset_index()

# 可视化柱状图
plt.figure(figsize=(10, 6))
plt.bar(classroom_summary['classroomId'], classroom_summary['noteNum'], label='笔记总数', color='skyblue', width=0.4)
plt.bar(classroom_summary['classroomId'] + 0.4, classroom_summary['threadNum'], label='话题总数', color='orange', width=0.4)
plt.xlabel('班级ID')
plt.ylabel('数量')
plt.title('每个班级的笔记数和话题数总和')
plt.legend()
plt.grid(True)
plt.show()
