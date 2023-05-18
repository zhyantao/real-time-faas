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

# 连接到 OpenWhisk 实例
wsk property set --apihost 192.168.163.146:31001
wsk property set --auth 23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP
wsk -i package list /whisk.system

# # (不可用) 暴露 prometheus 服务
# kubectl expose deployment -n openwhisk owdev-prometheus-server --type=NodePort --name=owdev-prometheus-ui # 通过公网 IP 访问
# kubectl port-forward -n openwhisk svc/owdev-prometheus-ui 31119:9090 &                                    # 通过内网 IP 访问

# 将 prometheus 采集的性能指标用 grafana 可视化
kubectl expose deployment -n openwhisk owdev-grafana --type NodePort --name=grafana # 使用公网 IP 访问
kubectl port-forward -n openwhisk svc/grafana 3000:3000 &
#GRAFANA_PORT=$(kubectl -n openwhisk get svc grafana -o jsonpath="{.spec.ports[0].nodePort}")
#GRAFANA_URL=http://192.168.163.146:$GRAFANA_PORT/monitoring

# 查看 openwhisk 命名空间下的公共 IP 和端口号
kubectl get svc -n openwhisk -o wide
