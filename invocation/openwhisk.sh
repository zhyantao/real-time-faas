# 连接到 OpenWhisk 实例
wsk property set --apihost 192.168.163.146:31001
wsk property set --auth 23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP
wsk -i package list /whisk.system

# 查看已部署的函数
wsk -i action list

# 删除函数
wsk -i action delete lin_pack

# 打包 functions
cd lin_pack/ # 一定要切换到这里面，因为相对路径会导致一些问题
zip -r lin_pack.zip *

# 创建 OpenWhisk 操作：入口函数的文件名必须为 __main__.py，且该文件中包含 main() 函数
# 使用本地依赖 (如: numpy) 创建函数
wsk -i action update lin_pack lin_pack.zip --docker 192.168.163.146:5000/python3action:1.1.0

# 调用 OpenWhisk 操作
wsk -i action invoke lin_pack --result
# wsk -i activation list # 报错时查看 activation id
# wsk -i activation logs <activation_id>