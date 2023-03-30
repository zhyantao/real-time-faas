from greet import say_hello_to

cold = True

def main(args):
    global cold
    was_cold = cold
    cold = False
    name = args.get("name", "stranger")
    greeting = say_hello_to(name)
    i = 0
    while i < 10000000:
        i +=1
    return {"greeting": greeting, "cold":was_cold}
