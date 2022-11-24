cat hosts | while read n; do
  node_name=$(echo $n | awk '{print $1}')
  ip=$(echo $n | awk '{print $2}')

  for node in $ip; do
    tar -zcP --file=./tmp/$node_name.tar.gz ./tmp/
    cp ./tmp/$node_name.tar.gz ./perf_data/$1
    echo "$node perf saved"
  done
done