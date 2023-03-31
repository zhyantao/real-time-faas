# 使用当前目录下的 Dockerfile 创建名为 test_env 的镜像
docker build -t python3action:1.0.0 .

# 将镜像标记为推送到本地仓库
docker tag python3action:1.0.0 192.168.163.146:5000/python3action:1.0.0

# 将创建好的镜像添加到本地 Docker 仓库
docker push 192.168.163.146:5000/python3action:1.0.0

# # (可选) 修改镜像 (如: 在镜像中继续安装其他依赖) 但是修改后，退出容器先前的设置就没有了
# docker run -it 192.168.163.146:5000/python3action:1.0.0 /bin/bash # 进入容器
# pip install boto3 uuid # 修改容器
# exit # 退出容器
# docker tag 192.168.163.146:5000/python3action:1.0.0 192.168.163.146:5000/python3action:2.0.0 # 提交修改到本地 registry 仓库
# docker push 192.168.163.146:5000/python3action:2.0.0 # 持久化到本地仓库
