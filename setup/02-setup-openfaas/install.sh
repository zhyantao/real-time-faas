# 以下内容在 master 节点上运行

# 安装 faas-cli 和 helm
curl -sSL https://cli.openfaas.com | sudo -E sh
curl -sSLf https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash

# 添加 OpenFaaS 仓库
helm repo add openfaas https://openfaas.github.io/faas-netes/
helm repo update \
 && helm upgrade openfaas --install openfaas/openfaas \
    --namespace openfaas  \
    --set functionNamespace=openfaas-fn \
    --set generateBasicAuth=true

# 安装 FaaS 服务
git clone https://github.com/openfaas/faas-netes.git
cd faas-netes && kubectl apply -f namespaces.yml
kubectl -n openfaas create secret generic basic-auth \
    --from-literal=basic-auth-user=admin \
    --from-literal=basic-auth-password=admin
kubectl apply -f ./yaml/

# 验证安装是否成功
watch -n 1 kubectl get pods -n openfaas

# 测试环境
export OPENFAAS_URL=http://127.0.0.1:31112
faas-cli login --password admin
faas-cli store deploy figlet
echo "hello, world!" | faas-cli invoke figlet

