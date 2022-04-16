class Student:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def say_age(self):
        print("{0}的年龄是{1}".format(self.name,self.age))
class Student1(Student):
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def say_age(self):
        print("{0}的年龄是{1}".format(self.name,self.age))
class Man:
    pass
s1=Student("2",23)
s1.say_age()