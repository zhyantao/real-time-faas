# 使用当前目录下的 Dockerfile 创建名为 test_env 的镜像
docker build -t python3action:1.1.1 .

# 将镜像标记为推送到本地仓库
docker tag python3action:1.1.1 192.168.163.146:5000/python3action:1.1.1

# 将创建好的镜像添加到本地 Docker 仓库
docker push 192.168.163.146:5000/python3action:1.1.1

# # (可选) 可进入容器运行代码，但是退出容器后，记录都被删除
# docker run -it 192.168.163.146:5000/python3action:1.1.0 /bin/bash # 进入容器
# pip install boto3 uuid # 运行代码
# exit # 退出容器