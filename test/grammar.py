def testForLoop():
    ans = [True for _ in range(1, 5)]
    print(ans)


def testAny():
    # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，是则返回 False
    # 如果有一个为 True，则返回 True。
    ans = any([True for _ in range(1, 5)])
    print(ans)


# if not：如果 statement 等于 空 或 false，返回 true
def testIf():
    # https://pythonexamples.org/python-if-not/
    a = False
    if not a:
        print('a is false.')

    string_1 = ''
    if not string_1:
        print('String is empty.')
    else:
        print(string_1)

    a = []
    if not a:
        print('List is empty.')
    else:
        print(a)

    a = dict({})
    if not a:
        print('Dictionary is empty.')
    else:
        print(a)

    a = set({})
    if not a:
        print('Set is empty.')
    else:
        print(a)

    a = tuple()
    if not a:
        print('Tuple is empty.')
    else:
        print(a)

    arr = [index for index in range(1, 5) if False]
    print(arr)
