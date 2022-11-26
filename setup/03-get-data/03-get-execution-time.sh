# 使用 perf 命令采集进程运行时的信息
mkdir -p tmp && rm tmp/* -rf

event=${1:-'LLC-load-misses'}
interval=${2:-'500'}
namespace=${3:-'openfaas-fn'}
kubectl get node -o wide | grep -v "NAME" | awk '{print $1,$6}' >hosts
timestamp=$(date +%s.%N)

cat hosts | while read n; do
  echo "handling ------- $n"
  hostname=$(echo $n | awk '{print $1}')
  ip=$(echo $n | awk '{print $2}')
  echo $hostname' - '$ip' - '$event
  for pod_name in $(kubectl get pod -o wide -n $namespace | grep $hostname | awk '{print $1}'); do
    echo 'pod_name' $pod_name ' in node' $hostname 2>&1 &
    cmd='bash get_container_pid.sh '$pod_name
    pid=$(echo $cmd | tr -cd "[0-9]" 2>&1 &)
    echo $pod_name' --- pid '$pid
    perf stat -e $event -I $interval -p $pid -o "./tmp/"$hostname'_'$pod_name'_'$timestamp".csv" 2>&1 &
  done
done
