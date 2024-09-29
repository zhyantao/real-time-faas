# Research On Real-Time FaaS Resource Allocation and Task Scheduling Methods

## Usage

```bash
# Clone the repository
git clone --recursive https://github.com/zhyantao/real-time-faas.git
# git submodule update --init

# Setup Python 3.11 and pip
sudo add-apt-repository ppa:deadsnakes/ppa
PYTHON_VERSION=python3.11
sudo apt install $PYTHON_VERSION $PYTHON_VERSION-dev \
  $PYTHON_VERSION-venv $PYTHON_VERSION-distutils \
  $PYTHON_VERSION-lib2to3 $PYTHON_VERSION-gdbm \
  $PYTHON_VERSION-tk
mkdir -p ~/venv
$PYTHON_VERSION -m venv ~/venv/$PYTHON_VERSION --without-pip
echo "source ~/venv/$PYTHON_VERSION/bin/activate" >> ~/.bashrc
source ~/.bashrc
curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_VERSION

# Install Graphviz and dependencies
sudo apt install graphviz graphviz-dev
pip install -r requirements.txt

# Setup PYTHONPATH
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

# Install Docker: https://getstarted.readthedocs.io/cli/docker.html
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates \
  curl software-properties-common \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl enable containerd

# Run tests
cd real-time-faas
python models/test/generate_dataset.py
python models/autoscaler/main.py
python models/test/test_autoscaler.py
python models/scheduler/main.py
python models/scheduler/dqn.py
python models/scheduler/dqn_v2.py
python models/scheduler/dqn_v3.py
python models/scheduler/ppo.py
python models/scheduler/loader.py
python models/scheduler/gcn.py
python models/test/test_figure.py
```
