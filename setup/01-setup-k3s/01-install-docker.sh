# 每台主机的名字不同
hostnamectl set-hostname k8s-master # 在主机 1 上操作
hostnamectl set-hostname k8s-worker1 # 在主机 2 上操作
hostnamectl set-hostname k8s-worker2 # 在主机 3 上操作

# 升级 Linux 内核到 5.4
rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
yum -y install https://www.elrepo.org/elrepo-release-7.0-4.el7.elrepo.noarch.rpm
yum --enablerepo="elrepo-kernel" -y install kernel-lt.x86_64
grub2-set-default 0
grub2-mkconfig -o /boot/grub2/grub.cfg
reboot
uname -r

# 同步时间
sudo systemctl stop chronyd
sudo systemctl disable chronyd
sudo yum install ntp ntpdate
sudo ntpdate ntp.aliyun.com
sudo rm -rf /etc/localtime
sudo ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 在 CentOS 7 上安装 Python 3.10
# 参考 https://docs.python.org/3/using/unix.html#custom-openssl 
# 确保存在 /etc/ssl
find /etc/ -name openssl.cnf -printf "%h\n"
# 安装 OpenSSL
curl -O https://www.openssl.org/source/openssl-VERSION.tar.gz
tar xzf openssl-VERSION
pushd openssl-VERSION
./config \
    --prefix=/usr/local \
    --libdir=lib \
    --openssldir=/etc/ssl
make -j1 depend
make -j8
make install_sw
popd
# 安装 Python 3
pushd python-3.x.x
./configure -C \
    --with-openssl=/usr/local \
    --with-openssl-rpath=auto \
    --prefix=/usr/local
make -j8
make altinstall
popd

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

vim /etc/rancher/k3s/registries.yaml
# 在 mirrors 下添加如下内容
mirrors:
  192.168.163.146:5000:
    endpoint:
      - "http://192.168.163.146:5000"

vim /etc/docker/daemon.json
# 在 daemon.json 中添加如下内容（没有文件则新建）
{
  "insecure-registries": [
    "192.168.163.146:5000"
  ]
}
systemctl restart containerd

# 在 master 节点上重启 k3s
systemctl restart k3s

# 在 node 节点上重启
systemctl restart k3s-agent

# 检查是否已经成功
cat  /var/lib/rancher/k3s/agent/etc/containerd/config.toml

# SSH 免密登录
ssh-keygen -t rsa # 生成公钥，在所有节点上都要运行
ssh-copy-id localhost # 发送私钥给本机，所有节点都要运行
ssh-copy-id <ip_address> # 发送公钥给其他机器，需要输入其他机器的密码