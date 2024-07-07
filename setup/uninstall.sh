# Undeploy OpenFaaS
kubectl delete -f faas-netes/yaml/
kubectl delete -f faas-netes/namespaces.yml

# Uninstall OpenWhisk
helm uninstall owdev -n openwhisk
