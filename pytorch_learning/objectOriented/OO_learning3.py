class Student:
    score=11
    company="lanzhou university"
    def __init__(self,name):
        self.name=name
    @classmethod
    def say_company(cls,name):
        print("{0}的公司是{1}".format(name,cls.company))
    @staticmethod
    def add(a,b):
        print("{}+{}={}".format(a,b,(a+b)))
        return Student.company
s1=Student("2",23)
s1.say_age()