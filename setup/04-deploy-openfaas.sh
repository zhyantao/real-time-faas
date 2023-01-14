# 该文件中的内容仅需在 master 节点运行
cd ../third_party/faas-netes/

# 设置系统环境变量
echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >>/etc/profile
source /etc/profile

# 安装 faas-cli 和 helm
curl -sSL https://cli.openfaas.com | sudo -E sh
bash get_helm.sh

# 部署 OpenFaaS 服务
helm repo add openfaas https://openfaas.github.io/faas-netes/

kubectl apply -f faas-netes/namespaces.yml

kubectl -n openfaas create secret generic basic-auth \
    --from-literal=basic-auth-user=admin \
    --from-literal=basic-auth-password="admin"

kubectl apply -f faas-netes/yaml/

# 验证服务是否安装成功
kubectl get pods -n openfaas

# 登录平台，并部署服务
echo "export OPENFAAS_URL=http://192.168.163.146:31112" >> ~/.bashrc
source ~/.bashrc
faas-cli login --password admin
faas-cli store deploy figlet
echo "hello, world" | faas-cli invoke figlet

# 暴露 prometheus 服务
kubectl expose deployment -n openfaas prometheus --type=NodePort --name=prometheus-ui # 通过公网 IP 访问
kubectl port-forward -n openfaas svc/prometheus-ui 31119:9090 &                       # 通过内网 IP 访问

# 将 prometheus 采集的性能指标用 grafana 可视化
# 参考 https://github.com/openfaas/workshop/blob/master/lab2.md
kubectl run -n openfaas grafana --image=stefanprodan/faas-grafana:4.6.3 --port=3000
kubectl expose pod -n openfaas grafana --type=NodePort --name=grafana # 使用公网 IP 访问
kubectl port-forward -n openfaas svc/grafana 3000:3000 &
#GRAFANA_PORT=$(kubectl -n openfaas get svc grafana -o jsonpath="{.spec.ports[0].nodePort}")
#GRAFANA_URL=http://192.168.163.146:$GRAFANA_PORT/dashboard/db/openfaas

# 查看 openfaas 命名空间下的公共 IP 和端口号
kubectl get svc -n openfaas -o wide

# 拉取 OpenFaaS templates
faas-cli template pull
