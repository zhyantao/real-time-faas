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

# Deploy OpenWhisk function
wsk -i action create hello /root/real-time-faas/functions/openwhisk/hello.js
