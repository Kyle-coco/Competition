
# 在线教育综合大数据分析

## 1. 项目概述

随着互联网和移动技术的普及，数字化教育已成为全球教育发展的重要趋势。本项目旨在设计和开发一个基于Web的在线教育综合大数据分析系统，通过对在线教育平台数据的深入分析，为教育机构提供辅助决策支持。该系统能够：

* **为讲师提供课程质量反馈**：帮助讲师了解课程效果，从而提升教学质量。
* **帮助班主任掌握学生学习情况**：实时监控班级学生的学习进度，进行预警，并支持调整教学策略和定制个性化学习方案。
* **优化教育资源配置**：根据线上课程使用情况，优化平台资源配置，提供更精准有效的教育服务。

最终目标是打造一个全面的在线教育平台，实现个性化学习和终身教育的可能。

## 2. 数据集

本项目分析涉及以下主要数据集：

* `activity_learn_log.csv`: 学习活动日志，包含用户学习时间等信息。
* `student_course_participation_counts.csv`: 学生课程参与度计数。
* `user_learn_statistics_total.csv`: 用户学习统计总览，包含学习时长和完成任务数。
* `classroom_member.csv`: 班级成员信息，包含用户ID、班级ID和角色。
* `log.csv`: 用户行为日志，包含各类用户操作及其时间戳。

## 3. 分析模块与功能及实验结果

本项目包含以下核心数据分析模块，并通过Python脚本实现。

### 3.1. 学生学习时长分布分析 (`studying_time_test.py`)

* **功能**: 计算并可视化学生总学习时长的分布情况，帮助了解学生的学习投入度。
* **实现细节**:
    * 合并 `user_learn_df` 和 `activity_log_df`。
    * 清洗数据，移除 `learnedTime` 列的缺失值。
    * 将学习时长从秒转换为小时。
    * 统计不同学习时长区间的用户数量，并生成柱状图展示。
* **结果示例**:
    * **学生学习时长分布图**
        [学生学习时长分布图](https://github.com/Kyle-coco/Competition/blob/main/TD/PHOTO_result/stt.png)
       

### 3.2. 学生学习速率分析 (`homework_speed.py`)

* **功能**: 计算每个学生的学习速率（每秒完成的任务数），并进行排序和导出。
* **实现细节**:
    * 加载 `user_learn_statistics_total.csv`。
    * 处理缺失值并确保 `learnedSeconds` 不为零。
    * 计算 `速率` = `finishedTaskNum` / `learnedSeconds`。
    * 按速率降序排序，并绘制柱状图展示。
    * 脚本还支持将排序后的学生速率数据输出到 `analyzed_student_data1.csv`。
* **结果示例**:
    * **学生完成课程材料速率排名图**
        [学生完成课程材料速率图](https://github.com/Kyle-coco/Competition/blob/main/TD/PHOTO_result/HS.png)
        

### 3.3. 班级任务完成速率分析 (`homework_speed2.py`)

* **功能**: 分析每个班级的平均任务完成速率（每小时完成的任务数），评估班级整体学习效率。
* **实现细节**:
    * 合并 `user_learn_statistics_total.csv` 和 `classroom_member.csv`。
    * 将学习时长转换为小时。
    * 按 `classroomId` 分组，计算每个班级的总任务完成数除以总学习小时数，得到每小时任务完成率。
    * 绘制散点图展示各班级的任务完成率。
* **结果示例**:
    * **各班级任务完成速率散点图**
        [班级任务完成速率散点图](https://github.com/Kyle-coco/Competition/blob/main/TD/PHOTO_result/HS2.png)
        

### 3.4. 学生参与度得分分析 (`Paticipation_test.py`)

* **功能**: 根据用户的行为日志计算学生的参与度得分，并分析其分布，以识别高参与度和低参与度的学生。
* **实现细节**:
    * 定义不同用户行为的权重（如 `login_success`, `create_course`, `create_thread` 等）。
    * 筛选出学生用户ID。
    * 根据预设权重计算每个学生的总参与度得分。
    * 将学生参与度得分导出到 `student_engagement_scores.csv`。
    * 绘制学生参与度得分的直方图分布。
* **结果示例**:
    * **学生参与度得分直方图**
        [学生参与度得分直方图](https://github.com/Kyle-coco/Competition/blob/main/TD/PHOTO_result/PT.png)
       

### 3.5. 笔记和话题数量分析 (`NoteNum.py`)

* **功能**: 统计和可视化每个班级的笔记总数和话题总数，反映班级内的知识共享和互动活跃度。
* **实现细节**:
    * 读取 `classroom_member.csv`。
    * 按 `classroomId` 分组，计算 `noteNum` 和 `threadNum` 的总和。
    * 绘制堆叠柱状图。
* **结果示例**:
    * **每个班级的笔记数和话题数总和图**
        [班级笔记与话题总数图](https://github.com/Kyle-coco/Competition/blob/main/TD/PHOTO_result/NN.png)
        **此图表明了每个班级的同学极其不喜欢做笔记和参与讨论**
        

## 4. 技术栈

* **数据分析与处理**: Pandas
* **数据可视化**: Matplotlib, Seaborn
* **编程语言**: Python

## 5. 系统架构设计 (Web部分待补充)

本项目的核心是后端数据分析，未来将扩展为Web系统，其架构初步设想如下：

* **数据层**: 存储原始数据和分析结果的数据库（例如：MySQL, PostgreSQL）。
* **分析层**: 基于Python和Pandas/Numpy的数据分析脚本，用于处理数据并生成洞察。
* **API层**: 提供RESTful API接口，供前端调用获取分析结果。
* **前端层**: 基于Web框架（如React, Vue.js, Angular）构建用户界面，可视化数据分析结果，并提供交互式报表。
