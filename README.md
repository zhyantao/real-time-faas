# 实时 FaaS 的资源调度与任务分配方法研究

## 使用方法

### 克隆仓库到本地

```bash
git clone git@gitee.com:zhyantao/real-time-faas.git
cd real-time-faas
git submodule update --init --recursive
```

### 环境配置

#### Windows

- 下载并安装 [Python 3.9.13](https://www.python.org/downloads/release/python-3913/)
- 下载并安装 [Graphviz 7.0.6](https://graphviz.org/download/)（添加至系统环境变量）
- 下载并安装 [PyGraphviz 1.10](https://pygraphviz.github.io/documentation/stable/install.html#manual-download)
- 安装依赖 `pip install -r requirements.txt`
- 用 PyCharm 打开项目，添加解释器，选择系统解释器
- 将 `real-time-faas` 添加至环境变量 `PYTHONPATH`

> ModuleNotFoundError: https://www.cnblogs.com/hi3254014978/p/15202910.html

#### CentOS

> 注意：CentOS 中的配置如非特殊说明一般是在 非 root 模式下运行。

```bash
cd real-time-faas

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
# 安装 Python 3.10.9
curl -O https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
tar xzf Python-3.10.9.tgz
pushd Python-3.10.9
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

echo 'export PYTHONPATH='"$PWD" >> /etc/profile
source /etc/profile

sudo yum install graphviz graphviz-devel
pip3 install --global-option=build_ext \
             --global-option="-I/usr/include/graphviz/" \
             --global-option="-L/usr/lib64/graphviz/" \
             pygraphviz
pip3 install -r requirements.txt

# 仅在 root 模式下运行上面的命令才需要运行下面的命令（不推荐使用 root 模式运行代码）
# echo "export PATH=$PATH:~/.local/bin" >> /etc/profile
# source /etc/profile
# jupyter notebook --generate-config
# echo "c.NotebookApp.allow_root=True" >> ~/.jupyter/jupyter_notebook_config.py
```

## 命名规范

C/C++ 和 Python 类名为驼峰式，函数名和普通变量为下划线式。

Java 和 Go 中的类名、函数名和普通变量都是驼峰式。
