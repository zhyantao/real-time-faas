# Deploy OpenFaaS function
OpenFaaS_FUNC_PATH='/root/real-time-faas/functions/openfaas'
OpenWhisk_FUNC_PATH='/root/real-time-faas/functions/openwhisk'

cd $OpenFaaS_FUNC_PATH || exit
faas-cli up -f openfaas.yml

# Deploy OpenWhisk function
#wsk -i action create hello $OpenWhisk_FUNC_PATH/hello.js
wsk -i action update hello $OpenWhisk_FUNC_PATH/hello.js
