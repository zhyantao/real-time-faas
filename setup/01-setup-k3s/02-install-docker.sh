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
sudo yum install docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker

# 测试 Docker 是否安装成功
sudo docker run hello-world

# 创建私有仓库，在每一台节点上运行
# 参考 https://www.yuque.com/wukong-zorrm/qdoy5p/gdxrr5si4vhkqia5
# 默认本地仓库存放路径为 /var/lib/registry
docker run -d -p 5000:5000 --restart=always --name registry registry
