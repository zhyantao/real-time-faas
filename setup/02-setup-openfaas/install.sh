# 添加系统环境变量
echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> /etc/profile
source /etc/profile

### 以下内容在 master 节点上运行

# 安装 faas-cli 和 helm
curl -sSL https://cli.openfaas.com | sudo -E sh
bash get_helm.sh

# 使用 Helm 安装 OpenFaaS 服务
echo Installing with helm 👑

helm repo add openfaas https://openfaas.github.io/faas-netes/

kubectl apply -f namespaces.yml
kubectl apply -f ./yaml/

PASSWORD=admin # 设置密码

kubectl -n openfaas create secret generic basic-auth \
    --from-literal=basic-auth-user=admin \
    --from-literal=basic-auth-password="$PASSWORD"

echo "Installing chart 🍻"
helm upgrade \
    --install \
    openfaas \
    openfaas/openfaas \
    --namespace openfaas  \
    --set basic_auth=true \
    --set functionNamespace=openfaas-fn \
    --set serviceType=LoadBalancer \
    --wait

# 验证安装是否成功
kubectl get pods -n openfaas

# 测试环境
export OPENFAAS_URL=http://127.0.0.1:31112
faas-cli login --password admin
faas-cli store deploy figlet
echo "hello, world!" | faas-cli invoke figlet

