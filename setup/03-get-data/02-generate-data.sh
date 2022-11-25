#!/bin/bash

uuid=$(cat /sys/class/dmi/id/product_uuid)
platform=$(uname -m)
invokeTimes='3'            # 调用次数
function_name='hello-java' # 函数名称
params='PigAndDog'         # 参数列表

for i in $(seq 1 $invokeTimes); do
  invokeTime=$(date +%s.%N)
  responseBody=$(curl -s http://192.168.163.146:31112/function/$function_name.openfaas-fn)
  endTime=$(date +%s.%N)
  startTime=$(echo $responseBody | jq -r '.startTime')
  if [ ! $startTime ]; then
    startTime="''"
  fi
  echo "$uuid,$platform,$function_name,$invokeTime,$startTime,$endTime" >>delay_and_execution_time_$platform.csv
done
