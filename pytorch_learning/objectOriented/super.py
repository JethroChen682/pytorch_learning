class A:
    def say(self):
        print("type:",type(self))
class B(A):
    def said(self):
        super(B,self).say()
s=B()
print(s.said())