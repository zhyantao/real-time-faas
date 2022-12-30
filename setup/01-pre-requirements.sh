# 静态分配 IP 地址
vim /etc/sysconfig/network-scripts/ifcfg-ens33
BOOTPROTO="static"
IPADDR="192.168.163.146"
GATEWAY="192.168.163.2"
DNS1="119.29.29.29"
systemctl restart network

# 添加主机名和 IP 地址的映射
vim /etc/hosts
192.168.163.146   k8s-master
192.168.163.147   k8s-worker1
192.168.163.148   k8s-worker2

# 每台主机的名字不同
hostnamectl set-hostname k8s-master  # 在主机 1 上操作
hostnamectl set-hostname k8s-worker1 # 在主机 2 上操作
hostnamectl set-hostname k8s-worker2 # 在主机 3 上操作

## 升级 Linux 内核到 5.4
#rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
#yum -y install https://www.elrepo.org/elrepo-release-7.0-4.el7.elrepo.noarch.rpm
#yum --enablerepo="elrepo-kernel" -y install kernel-lt.x86_64
#grub2-set-default 0
#grub2-mkconfig -o /boot/grub2/grub.cfg
#reboot
## （谨慎操作）删除不需要的内核
#yum remove $(rpm -qa | grep kernel | grep -v $(uname -r))
#reboot

# 同步时间
sudo systemctl stop chronyd
sudo systemctl disable chronyd
sudo yum install ntp ntpdate
sudo ntpdate ntp.aliyun.com
sudo rm -rf /etc/localtime
sudo ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 在 CentOS 7 上安装 Python 3.10 和 Git 2.38.1
# 参考 https://docs.python.org/3/using/unix.html#custom-openssl
# 确保下面这条命令能搜索出结果
find /etc/ -name openssl.cnf -printf "%h\n"
# 安装 OpenSSL
curl -O https://www.openssl.org/source/openssl-1.1.1.tar.gz
tar xzf openssl-1.1.1.tar.gz
pushd openssl-1.1.1
./config \
    --prefix=/usr/local \
    --libdir=lib \
    --openssldir=$(find /etc/ -name openssl.cnf -printf "%h\n")
make -j1 depend
make -j8
make install_sw
popd
# 安装 Python 所需的依赖
yum update
yum install yum-utils
yum groupinstall "Development Tools"
yum install bzip2-devel ncurses-devel \
    gdbm-devel \
    sqlite-devel tk-devel libuuid-devel \
    readline-devel zlib-devel \
    libpcap-devel xz-devel expat-devel libffi libffi-devel
# 安装 Python 3.10.7
curl -O https://www.python.org/ftp/python/3.10.7/Python-3.10.7.tgz
tar xzf Python-3.10.7.tgz
pushd Python-3.10.7
./configure -C \
    --with-openssl=/usr/local \
    --with-openssl-rpath=auto \
    --prefix=/usr/local
make -j8
make altinstall
popd
rm -rf /usr/local/bin/python3 /usr/local/bin/pip3
ln -s /usr/local/bin/python3.10 /usr/local/bin/python3
ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip3
# 安装 Git 2.38.1
yum -y install libcurl-devel
curl -O https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.38.1.tar.gz
tar xzf git-2.38.1.tar.gz
pushd git-2.38.1
./configure -C \
    --with-openssl=/usr/local \
    --with-openssl-rpath=auto \
    --prefix=/usr/local
make -j8
make install
popd

# 安装 Go
curl -O https://go.dev/dl/go1.19.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.19.4.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >>/etc/profile
source /etc/profile
# 设置 Go 代理
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct

# SSH 免密登录
ssh-keygen -t rsa       # 生成公钥，在所有节点上都要运行
ssh-copy-id k8s-master  # 发送公钥给其他机器，需要输入其他机器的密码
ssh-copy-id k8s-worker1 # 发送公钥给其他机器，需要输入其他机器的密码
ssh-copy-id k8s-worker2 # 发送公钥给其他机器，需要输入其他机器的密码

# 安装性能监测工具 perf
yum install perf

# seekwatcher 依赖于 Python 3.10，安装 pip
# yum install python2-devel    # for python2.x installs
# curl -O https://bootstrap.pypa.io/pip/2.7/get-pip.py
# python get-pip.py install
pip3 install cython
yum install python3-tkinter
# 安装 seekwatcher
curl -O https://github.com/trofi/seekwatcher/archive/refs/tags/v0.14.tar.gz
tar zxf seekwatcher-0.14.tar.gz && cd seekwatcher-0.14/
vim cmd/seekwatcher # 将第一行的 python 修改为 python3
python3 setup.py install

# 授权 AWS 命令行访问（不安全，而且会产生费用）
echo 'export AWS_ACCESS_KEY_ID=' >> ~/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY=' >> ~/.bashrc
source ~/.bashrc
