set -eu

if [[ "$#" -ne 1 ]]
then
    echo "Usage: ./install_python.sh gpu"
    exit 1
fi

python3 -m pip install -U pip
pip3 install setuptools>=41.0.0
if [ $1 == "gpu" ]; then
    echo "Install GPU"
    pip3 install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

    pip3 install -U numpy wheel ninja
    
    # Build MinkowskiEngine from source
    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    git checkout v0.5.4

    export CUDA_HOME=/usr/local/cuda
    export FORCE_CUDA=1
    python3 setup.py install
    cd ..

    git clone https://github.com/mit-han-lab/torchsparse.git
    cd torchsparse
    git checkout v1.4.0
    export CUDA_HOME=/usr/local/cuda
    python setup.py install
    cd ..

    pip3 install pycuda
else
    echo "Install CPU"
    pip3 install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install MinkowskiEngine
    pip3 install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
fi
