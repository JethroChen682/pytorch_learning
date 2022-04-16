# __call__ TEST
class Student:
    def work(self):
        print("{} is an address ".format(self))
def work2(s):
    print("{} is also an address".format(s))
s1=Student()
s1.work()
Student.work=work2
s1.work()