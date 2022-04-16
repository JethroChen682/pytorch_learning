# 析构方法
class Student:
    def __del__(self):
        print("销毁对象{}".format(self))
s1=Student()
s2=Student()
del s2
print("program is over")