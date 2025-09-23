# initialise remote machine for testing purposes
# This script is intended to be run on a remote machine to set up the environment for testing
sudo apt update && sudo apt install -y vim unzip && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash ./Miniconda3-latest-Linux-x86_64.sh && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda init && touch ~/.no_auto_tmux && exit

conda create -n ML python==3.10 && conda activate ML && pip install numpy torch torchvision matplotlib pandas tensorboardx tqdm scikit-image scikit-learn cma-es && git clone "https://github.com/mahadevxo/Distributed-ViT.git" && sudo apt install -y nvtop && conda deactivate && pip install nvitop && conda activate ML && pip install nvitop
