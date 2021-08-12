#! /bin/bash

git lfs install
git submodule update --init --progress --depth=1 --recursive
python3 -m pip install --user -r requirements.txt


# TODO: Suppory for more system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt install -y pandoc graphviz
else
    echo "Not supported for current system now."
fi

