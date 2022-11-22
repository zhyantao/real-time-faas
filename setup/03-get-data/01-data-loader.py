import os

#path_ml = '/root/real-time-faas/functions/machine-learning/'
#path_bigdata = '/root/real-time-faas/functions/big-data'
#path_stream = '/root/real-time-faas/functions/stream'
path_web = '/root/real-time-faas/functions/web-services'

def update_openfaas(path):
    os.chdir(path)
    current_path = os.getcwd()
    print(current_path)
    build = 'action_build.sh'
    r = os.popen('bash '+build).read()
    print(r)
    push = 'action_push.sh'
    r = os.popen('bash '+push).read()
    print(r)
    deploy = 'action_deploy.sh'
    r = os.popen('bash '+deploy).read()
    print(r)
    
#update_openfaas(path_ml)
#update_openfaas(path_bigdata)
#update_openfaas(path_stream)
update_openfaas(path_web)
