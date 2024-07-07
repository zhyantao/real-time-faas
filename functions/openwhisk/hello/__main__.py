from greet import say_hello_to
import sys

cold = True


# OpenWhisk 使用的环境和本地 Python 环境是不一样的
def main(args):
    global cold
    was_cold = cold
    cold = False
    name = args.get("name", "stranger")
    greeting = say_hello_to(name)
    return {
        "greeting": greeting, 
        "cold": was_cold,
        "当前 Python 版本：": sys.version,
        "当前 Python 主要版本号：": sys.version_info.major,
        "当前 Python 次要版本号：": sys.version_info.minor,
        "当前 Python 微版本号：": sys.version_info.micro
        }

if __name__ == '__main__':
    main({})
