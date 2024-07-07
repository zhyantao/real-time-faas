# Research On Real-Time FaaS Resource Allocation and Task Scheduling Methods

## Usage

```bash
# Clone the repository
git clone --recursive https://github.com/zhyantao/real-time-faas.git
# git submodule update --init

# Setup Python 3.11 and pip
sudo add-apt-repository ppa:deadsnakes/ppa
PYTHON_VERSION=python3.11
sudo apt install $PYTHON_VERSION $PYTHON_VERSION-dev
$PYTHON_VERSION -m venv $PYTHON_VERSION --without-pip
source $PYTHON_VERSION/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_VERSION
pip install -r requirements.txt

# Install Graphviz
sudo apt install graphviz graphviz-dev

# Setup PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

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
