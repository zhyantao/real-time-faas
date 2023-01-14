# 该文件中的内容仅需在 master 节点运行
cd ../third_party/openwhisk-deploy-kube/

kubectl label nodes --all openwhisk-role=invoker

curl -O https://github.com/apache/openwhisk-cli/releases/download/1.2.0/OpenWhisk_CLI-1.2.0-linux-amd64.tgz
tar -xvf OpenWhisk_CLI-1.2.0-linux-amd64.tgz
cp wsk /usr/local/bin/wsk

kubectl create namespace openwhisk
cd openwhisk-deploy-kube
cp ../../configs/openwhisk.yaml .
helm install owdev ./helm/openwhisk -n openwhisk --create-namespace -f openwhisk.yaml
watch -n 1 kubectl get pods -n openwhisk

wsk property set --apihost 192.168.163.146:31001
wsk property set --auth 23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP
wsk -i package list /whisk.system
