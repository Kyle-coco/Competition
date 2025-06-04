import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("D:/PyCharm/TD_data_analysis/game_TD/user_learn_statistics_total.csv")

# 检查数据完整性
if data.isnull().values.any():
    data = data.dropna()  # 删除缺失值

# 计算学生速率
data = data.dropna(subset=['learnedSeconds'])
data = data[data['learnedSeconds'] != 0]
data['learnedHours'] = data['learnedSeconds'] / 3600# 将学习时长从秒转换为小时
data['速率'] = data['finishedTaskNum'] / data['learnedSeconds']



# 对学生速率进行排序
sorted_data = data.sort_values(by='速率', ascending=False)

# 可视化学生速率
plt.rcParams['font.sans-serif'] = ['SimHei']#为了在图上显示中文
plt.rcParams["font.size"] = 20#调整字体大小
plt.rcParams['axes.unicode_minus'] = False#为了在图上显示负号

plt.figure(figsize=(10,6))
plt.figure(figsize=(12, 8), dpi=100)
plt.bar(sorted_data['userId'], sorted_data['速率'])
plt.xlabel('学生姓名')
plt.ylabel('速率')
plt.title('学生完成课程材料的速率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

将分析结果输出到文件
sorted_data.to_excel('analyzed_student_data1.xlsx', index=False)
result = data[['userId', '速率']]

output_csv_path = 'analyzed_student_data1.csv'
sorted_data.to_csv(output_csv_path, index=False)

# 显示分析结果
print(result)

