# [参考] https://github.com/apache/openwhisk/blob/master/docs/actions-docker.md
FROM openwhisk/python3action:latest

# 安装必要环境
RUN apk add jpeg-dev zlib-dev freetype-dev lcms2-dev openjpeg-dev tiff-dev tk-dev tcl-dev openblas-dev
# 安装 pip
RUN apk add --update py-pip

# 打印 Python 版本
RUN python --version

# 安装依赖
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY tensorflow-0.12.0-py3-none-any.whl .
RUN pip install tensorflow-0.12.0-py3-none-any.whl
