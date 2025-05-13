#! /usr/bin/env bash

/bin/bash /opt/conda/miniforge.sh -b -p /opt/install/miniconda

export PATH=$PATH:/opt/install/miniconda/bin

conda update --all --yes
conda clean -tipy

echo "auto_update_conda: False" >> /opt/install/miniconda/.condarc
echo "ssl_verify: False" >> /opt/install/miniconda/.condarc
echo ". /opt/install/miniconda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc
ln -s /opt/install/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
echo ". /opt/install/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc
echo "conda activate base" >> /root/.bashrc

conda install -k -y tini
chmod -R ugo+w /opt/install/miniconda
conda clean -tipy

conda create -n legere

echo "conda activate legere" >> /root/.bashrc

conda run --no-capture-out -n legere \
  conda install -c conda-forge -y \
    python==3.12 \
    cmake==3.31 \
    openssh

