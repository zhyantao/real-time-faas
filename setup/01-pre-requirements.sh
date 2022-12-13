# 每台主机的名字不同
hostnamectl set-hostname k8s-master  # 在主机 1 上操作
hostnamectl set-hostname k8s-worker1 # 在主机 2 上操作
hostnamectl set-hostname k8s-worker2 # 在主机 3 上操作

# 升级 Linux 内核到 5.4
rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
yum -y install https://www.elrepo.org/elrepo-release-7.0-4.el7.elrepo.noarch.rpm
yum --enablerepo="elrepo-kernel" -y install kernel-lt.x86_64
grub2-set-default 0
grub2-mkconfig -o /boot/grub2/grub.cfg
reboot
yum remove $(rpm -qa | grep kernel | grep -v $(uname -r)) # 删除不需要的内核
reboot

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

# 安装 Go
wget https://go.dev/dl/go1.19.3.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.19.3.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >>/etc/profile
source /etc/profile
# 设置 Go 代理
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct

# SSH 免密登录
ssh-keygen -t rsa     # 生成公钥，在所有节点上都要运行
ssh-copy-id localhost # 发送私钥给本机，所有节点都要运行
ssh-copy-id Remote-IP # 发送公钥给其他机器，需要输入其他机器的密码

# 安装性能监测工具 perf
yum install perf
