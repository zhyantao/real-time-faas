# 实时 FaaS 的资源调度与任务分配方法研究

## 使用方法

### 克隆仓库到本地

```bash
git clone git@gitee.com:zhyantao/real-time-faas.git
git submodule update --init --recursive
```

### 环境配置

#### Windows

- 下载并安装 [Python 3.10.9](https://www.python.org/downloads/release/python-3109/)
- 下载并安装 [Graphviz 7.0.6](https://graphviz.org/download/)（添加至系统环境变量）
- 下载并安装 [PyGraphviz 1.10](https://pygraphviz.github.io/documentation/stable/install.html#manual-download)
- 安装依赖 `pip install -r requirements.txt`
- 用 PyCharm 打开项目，添加解释器，选择系统解释器。

#### CentOS

```bash
#export PYTHONPATH="$PWD" # Linux
sudo yum install graphviz graphviz-devel
pip3 install pygraphviz
pip3 install -r requirements.txt
```
