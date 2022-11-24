# 使用 perf 命令采集进程运行时的信息
event=${1:-'LLC-load-misses'}
interval=${2:-'500'}
namespace=${3:-'openfaas-fn'}
kubectl get node -o wide | grep -v "NAME" | awk '{print $1,$6}' >hosts
TIMESTAMP=$(date +%s.%N)
cat hosts | while read n; do
  echo "handling $n"
  hostname=$(echo $n | awk '{print $1}')
  ip=$(echo $n | awk '{print $2}')
  #ssh -n $ip 'cd ./tmp/ && rm *.csv ' 2>&1 &
  echo $hostname' - '$ip' - '$event
  for pod in $(kubectl get pod -o wide -n $namespace | grep $hostname | awk '{print $1}'); do
    echo 'pod' $pod ' in node' $hostname 2>&1 &
    cmd='bash get_container_pid.sh '$pod
    pidstring=$cmd
    pid=$(echo $pidstring | tr -cd "[0-9]" 2>&1 &)
    echo 'pid '$pid
    perf stat -e $event -I $interval -p $pid -o "./tmp/"$hostname'_'$pod'_'$TIMESTAMP".csv" 2>&1 &
  done
done
