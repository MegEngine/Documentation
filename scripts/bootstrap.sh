#! /bin/bash
set -e

# TODO: Support for more systems and linux distributions
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt install -y git-lfs pandoc graphviz
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install git-lfs pandoc graphviz
else
    echo "Not support now."
fi

git lfs install
git submodule update --init --progress --depth=1 --recursive
python3 -m pip install --user -r requirements.txt


