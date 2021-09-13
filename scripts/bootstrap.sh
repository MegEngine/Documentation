#! /bin/bash
set -e

sudo apt install git-lfs
git lfs install
git submodule update --init --progress --depth=1 --recursive
python3 -m pip install --user -r requirements.txt

# TODO: Support for more systems and linux distributions
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt install -y pandoc graphviz
else
    echo "Not support now."
fi

