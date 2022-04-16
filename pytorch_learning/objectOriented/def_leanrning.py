# *args,**kwargs
def a2add(*num):
    num=num[0]
    if type(num)==tuple :
        print(type(num[len(num)-1]))
        sum=num[len(num)-1]**2+a2add(num[0:len(num)-1])
        return sum
    else:
        return num**2
c=a2add((1.0,2.0))