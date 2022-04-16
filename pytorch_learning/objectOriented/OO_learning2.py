class Student:
    score=11
    company="lanzhou university"
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def say_age(self):
        print("{0}的年龄是{1}".format(self.name,self.age))
    def say_company(self,company):
        print("{0}的公司是{1}".format(self.name,company))

s1=Student("2",23)
s1.say_age()