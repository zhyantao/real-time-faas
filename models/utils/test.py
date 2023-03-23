class A:
    def __init__(self):
        self.a = 0

    def func1(self):
        self.a += 1

    def func2(self):
        self.a += 1


if __name__ == '__main__':
    a = A()
    a.func1()
    a.func2()
    print(a.a)
