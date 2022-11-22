# 设置系统环境变量
echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> /etc/profile
source /etc/profile

# 安装 faas-cli 和 helm
curl -sSL https://cli.openfaas.com | sudo -E sh
bash get_helm.sh

# 部署 OpenFaaS 服务
helm repo add openfaas https://openfaas.github.io/faas-netes/

kubectl apply -f namespaces.yml

kubectl -n openfaas create secret generic basic-auth \
    --from-literal=basic-auth-user=admin \
    --from-literal=basic-auth-password="admin"

kubectl apply -f ./yaml/

# 验证服务是否安装成功
kubectl get pods -n openfaas

# 登录平台，并部署服务
export OPENFAAS_URL=http://192.168.163.146:31112
faas-cli login --password admin
faas-cli store deploy figlet
echo "hello, world" | faas-cli invoke figlet