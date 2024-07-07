# 删除旧版本的 Docker
sudo yum remove docker \
    docker-client \
    docker-client-latest \
    docker-common \
    docker-latest \
    docker-latest-logrotate \
    docker-logrotate \
    docker-engine

# 使用仓库安装 Docker
sudo yum install -y yum-utils device-mapper-persistent-data lvm2

sudo yum-config-manager \
    --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

# 更新系统软件仓库
sudo yum update

# 安装最新版的 Docker 引擎
sudo yum install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl enable containerd

# 测试 Docker 是否安装成功
sudo docker run hello-world

# 删除无效的镜像
docker image prune -f                                             # 清理 <none> 容器
docker rmi -f $(docker images | grep '<none>' | awk '{print $3}') # 清理 <none> 容器
docker rm $(docker ps -a | grep Exited | awk '{print $1}')        # 清理异常退出的容器

# 列出所有镜像
docker images

# 删除指定镜像
docker image rm 192.168.163.146:5000/python3action:1.0.0
docker image rm openwhisk/action-python-v3.7:1.17.0
