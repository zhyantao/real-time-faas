# 拉取源代码
faas-cli template pull
faas-cli new node12demo --lang node12

# 构建代码
faas-cli build -f node12demo.yml

# 将代码推送到本地仓库
faas-cli push -f node12demo.yml

# 发布到网页上
export OPENFAAS_URL=http://192.168.163.146:31112
faas-cli login --password admin
faas-cli deploy -f node12demo.yml