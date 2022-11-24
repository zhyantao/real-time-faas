#!/bin/bash
PRINTLOG=false
ACTIONNAME='hello-java' # 这个参数在 OpenFaaS 可视化界面可以看到
TIMES='100'             # 循环次数
PARAMS='PigAndDog'      # 函数需要的参数
PLATFORM=$(uname -m)
UUID=$(cat /sys/class/dmi/id/product_uuid)
while getopts "a:t:p:P:u:lWR" OPT; do
  case $OPT in
  a)
    ACTIONNAME=$OPTARG
    echo $ACTIONNAME
    ;;
  t)
    TIMES=$OPTARG
    ;;

  # "Warm up only" with this argument: warm up and then exit with no output.
  p)
    PARAMS=$(echo $OPTARG | sed $'s/\'//g')
    ;;
  P)
    PLATFORM=$OPTARG
    ;;
  u)
    UUID=$OPTARG
    ;;
  ?)
    echo "unknown arguments"
    ;;
  esac
done

if [[ -z $TIMES ]]; then
  TIMES=1
fi

LATENCYSUM=0
for i in $(seq 1 $TIMES); do
  invokeTime=$(date +%s.%N)
  times=$(curl -s http://192.168.163.146:31112/function/$ACTIONNAME.openfaas-fn)
  endTime=$(date +%s.%N)
  startTime=$(echo $times | jq -r '.startTime')
  if [ ! $startTime ]; then
    startTime="''"
  fi
  echo "$UUID,$PLATFORM,$ACTIONNAME,$invokeTime,$startTime,$endTime" >>result_$PLATFORM.csv
done
