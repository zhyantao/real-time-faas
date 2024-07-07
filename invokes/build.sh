# Deploy OpenFaaS function
OpenFaaS_FUNC_PATH='/home/user02/real-time-faas/functions/openfaas'
OpenWhisk_FUNC_PATH='/home/user02/real-time-faas/functions/openwhisk'

cd $OpenFaaS_FUNC_PATH || exit
faas-cli build -f openfaas.yml
faas-cli push -f openfaas.yml
faas-cli deploy -f openfaas.yml

# Deploy OpenWhisk function
wsk -i action update hello $OpenWhisk_FUNC_PATH/hello.js
