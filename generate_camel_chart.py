import matplotlib.pyplot as plt

# 数据
data = {'Stars': 12100, 'Forks': 1300}
labels = list(data.keys())
values = list(data.values())

# 创建条形图
fig, ax = plt.subplots()
bars = ax.bar(labels, values)

# 添加标题和标签
plt.title('GitHub Repository Statistics for camel-ai/camel')
plt.xlabel('Metric')
plt.ylabel('Count')

# 在条形上显示数值
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') # va: vertical alignment

# 保存图像
plt.savefig('camel_stats.png', bbox_inches='tight')

# 显示图表
plt.show()