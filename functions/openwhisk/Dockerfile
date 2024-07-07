# [参考] https://github.com/apache/openwhisk/blob/master/docs/actions-docker.md
FROM openwhisk/python3action:latest

# 安装必要环境
RUN apk add lcms2-dev openjpeg-dev tiff-dev tk-dev tcl-dev make cmake openblas-dev freetype-dev
RUN apk add g++ postgresql-dev cargo gcc python3-dev libffi-dev musl-dev zlib-dev jpeg-dev
# 安装 pip
RUN apk add --update py-pip

# 打印 Python 版本
RUN python --version

# 安装依赖
COPY tensorflow-0.12.0-py3-none-any.whl .
RUN pip install tensorflow-0.12.0-py3-none-any.whl

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
