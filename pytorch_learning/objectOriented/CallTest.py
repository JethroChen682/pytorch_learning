class photo:
    def __init__(self,ob):
        self.ob=ob
    def __call__(self,monthsalary):
        yearsalary=monthsalary*12
        return dict(year=yearsalary)
s=photo(3)
m=s(3000).items()