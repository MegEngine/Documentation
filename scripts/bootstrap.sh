#! /bin/bash
set -e

# TODO: Support for more systems and linux distributions
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y git-lfs pandoc graphviz doxygen libgl1-mesa-glx
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install git-lfs pandoc graphviz doxygen
else
    echo "Not support now."
fi

git lfs install
git lfs pull
git submodule update --init --progress --depth=1 --recursive
doxygen Doxyfile
python3 -m pip install --user -r requirements.txt

if [[ "$1" == megengine ]]; then
    echo "Install the latest MegEngine package..."
    python3 -m pip install megengine
else
    echo "If you want to build the complete documentation, please install the MegEngine package manually."
fi
