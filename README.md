# 实时 FaaS 的资源调度与任务分配方法研究

- CentOS 7 内核版本：5.4.224-1.el7.elrepo.x86_64 / Windows 11
- Python >= 3.8，将项目根目录添加到 Windows 环境变量 `PYTHONPATH` 中
- NVIDIA RTX 2060 专用 GPU 内存 6G
- 下载安装 [Graphviz](https://graphviz.org/) 并添加至系统环境变量

## 项目安装

```bash
export PYTHONPATH="$PWD" # Linux
pip3 install -r requirements.txt
git submodule update --init --recursive
yum install graphviz
```
