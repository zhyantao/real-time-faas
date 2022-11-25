# 获取某个容器的 pid
Check_jq() {
  which jq &>/dev/null
  if [ $? != 0 ]; then
    echo "no jq"
    $(yum install -y jq 2>/dev/null)
    exit 1
  fi
}
Pid_info() {
  docker_storage_location=$(docker info | grep 'Docker Root Dir' | awk '{print $NF}')
  for docker_short_id in $(docker ps | grep ${pod_name} | grep -v pause | awk '{print $1}'); do
    docker_long_id=$(docker inspect ${docker_short_id} | jq ".[0].Id" | tr -d '"')
    cat ${docker_storage_location}/containers/${docker_long_id}/config.v2.json | jq ".State.Pid"
  done
}
# 需要匹配的 Pod 名称，原来这个变量的值是 $1，不知道是否是错误的？
pod_name=$1
Check_jq
Pid_info
