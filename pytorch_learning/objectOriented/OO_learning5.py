# __call__ TEST
class Student:
    def __call__(self, salary):
        yearsalary=salary*12
        daysalary=salary//22.5
        hoursalary=daysalary/8
        return dict(yearsalary=yearsalary,daysalary=daysalary,hoursalary=hoursalary)
s1=Student()
s1(3500)