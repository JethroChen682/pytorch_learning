import matplotlib.pyplot as plt


# 同时打开两个窗口显示图片
plt.figure()  # 图片一
plt.imshow([[1]])
plt.figure()  # 图片二
plt.imshow([[2]])
# 显示前关掉交互模式
plt.ioff()
plt.show()