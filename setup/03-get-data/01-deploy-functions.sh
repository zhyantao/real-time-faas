# 登录到网页上
export OPENFAAS_URL=http://192.168.163.146:31112
faas-cli login --password admin

# Java function
cd /root/real-time-faas/functions/web-services/java-code
faas-cli build -f openfaas.yml
faas-cli push -f openfaas.yml
faas-cli deploy -f openfaas.yml

# NodeJS function
cd /root/real-time-faas/functions/web-services/nodejs-code
faas-cli build -f openfaas.yml
faas-cli push -f openfaas.yml
faas-cli deploy -f openfaas.yml

# Python function
cd /root/real-time-faas/functions/web-services/python-code
faas-cli build -f openfaas.yml
faas-cli push -f openfaas.yml
faas-cli deploy -f openfaas.yml
