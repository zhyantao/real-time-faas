# 关闭防火墙
systemctl disable firewalld --now

# 设置 SELinux
yum install -y container-selinux selinux-policy-base
yum install -y https://rpm.rancher.io/k3s/latest/common/centos/7/noarch/k3s-selinux-0.2-1.el7_8.noarch.rpm

# 下载并安装 K3s 和相关镜像文件
wget https://github.com/k3s-io/k3s/releases/download/v1.25.0%2Bk3s1/k3s
wget https://github.com/k3s-io/k3s/releases/download/v1.25.0%2Bk3s1/k3s-airgap-images-amd64.tar.gz
cp k3s /usr/local/bin
chmod +x /usr/local/bin/k3s
mkdir -p /var/lib/rancher/k3s/agent/images/
cp ./k3s-airgap-images-amd64.tar.gz /var/lib/rancher/k3s/agent/images/

# 在 k8s-master 节点上运行
git clone git@gitee.com:zhyantao/real-time-faas.git && cd real-time-faas/setup
chmod +x install.sh                         # 修改权限
scp install.sh root@k8s-worker1:~/tmp
scp install.sh root@k8s-worker2:~/tmp
INSTALL_K3S_SKIP_DOWNLOAD=true ./install.sh # 离线安装
kubectl get node                            # 安装完成后，查看节点状态
cat /var/lib/rancher/k3s/server/node-token  # 查看token
#K10c4b79481685b50e4bca2513078f4e83b62d1d0b5f133a8a668b65c8f9249c53e::server:bf7b63be7f3471838cbafa12c1a1964d

# 在 k8s-worker1 和 k8s-worker2 上运行
INSTALL_K3S_SKIP_DOWNLOAD=true \
  K3S_URL=https://192.168.163.146:6443 \
  K3S_TOKEN=K1012bdc3ffe7a5d89ecb125e56c38f9fe84a9f9aed6db605f7698fa744f2f2f12f::server:fdf33f4921dd607cadf2ae3c8eaf6ad9 \
  ./install.sh

# 创建私有仓库，在每一台节点上运行
# 参考 https://www.yuque.com/wukong-zorrm/qdoy5p/gdxrr5si4vhkqia5
# 默认本地仓库存放路径为 /var/lib/registry
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker run -d -p 5000:5000 --restart always --name registry registry:2

# 编辑 /etc/rancher/k3s/registries.yaml 添加如下内容（所有节点，没有则新建）
mirrors:
  docker.io:
    endpoint:
      - "https://fsp2sfpr.mirror.aliyuncs.com/"
  192.168.163.146:5000:
    endpoint:
      - "http://192.168.163.146:5000"
# 编辑 /etc/docker/daemon.json 添加如下内容（没有文件则新建）
{
  "insecure-registries": ["192.168.163.146:5000"]
}

# 重启服务
systemctl restart containerd # 在所有节点上重启服务
systemctl restart k3s        # 在 master 节点上重启服务
systemctl restart k3s-agent  # 在 node 节点上重启服务

# 检查是否已经成功
cat /var/lib/rancher/k3s/agent/etc/containerd/config.toml
