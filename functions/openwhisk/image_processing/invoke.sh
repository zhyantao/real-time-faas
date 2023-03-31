function_name=$(basename $PWD)
zip_name=$function_name.zip
rm -rf *.zip
zip -r $zip_name *
wsk -i action update $function_name $zip_name --docker 192.168.163.146:5000/python3action:1.1.1
wsk -i action invoke $function_name --result