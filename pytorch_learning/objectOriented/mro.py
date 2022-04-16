class A:
    def say(self):
        print("A")
class B(A):
    def say(self):
        print("B")
class C(A):
    def say(self):
        print("C")
def manystatus(m):
    if isinstance(m,A):
        m.say()
manystatus(A())