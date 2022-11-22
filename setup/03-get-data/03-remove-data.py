action_names = ['hello-python', 'hash-python', 
                'cryptography-python', 'md5-python', 'sort-python']
for i in action_names:
    cmd = 'faas-cli remove {function}'
    os.system(cmd.format(function=i))
