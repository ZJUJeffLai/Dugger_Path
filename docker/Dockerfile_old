#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
CMD nvidia-smi

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  wget \
  git \
  curl \
  ca-certificates \
  libjpeg-dev \
  libpng-dev \
  vim \
  apt-utils \
  software-properties-common \
  emacs24-nox \
  net-tools \
  && rm -rf /var/lib/apt/lists/*

######################################
# SECTION 1: Python 3.6 packages     #
######################################

# Outdated:
# RUN add-apt-repository ppa:jonathonf/python-3.6 && apt-get update \
#   && apt-get install -y python3.6 python3.6-dev python3.6-venv

# Python 3.6 Dependencies:
RUN apt-cache policy sqlite3
RUN apt-get install -y make 
#     libssl-dev \
#     zlib1g-dev \
#     libbz2-dev \
#     libreadline-dev \
#     libsqlite3-dev
#     llvm \
#     libncurses5-dev \
#     libncursesw5-dev \
#     xz-utils \
#     tk-dev
#     #build-essential \
#     #wget \
#     #curl \
# RUN apt-get -y install openssl
RUN apt-get update && apt-get install -y autoconf

RUN wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl-dev_1.0.2g-1ubuntu4.17_amd64.deb 
RUN apt install -y ./libssl-dev_1.0.2g-1ubuntu4.17_amd64.deb \
    && rm ./libssl-dev_1.0.2g-1ubuntu4.17_amd64.deb

RUN wget http://security.ubuntu.com/ubuntu/pool/main/z/zlib/zlib1g-dev_1.2.8.dfsg-2ubuntu4.3_amd64.deb 
RUN apt install -y ./zlib1g-dev_1.2.8.dfsg-2ubuntu4.3_amd64.deb \
    && rm ./zlib1g-dev_1.2.8.dfsg-2ubuntu4.3_amd64.deb

RUN wget http://security.ubuntu.com/ubuntu/pool/main/b/bzip2/libbz2-dev_1.0.6-8ubuntu0.2_amd64.deb 
RUN apt install -y ./libbz2-dev_1.0.6-8ubuntu0.2_amd64.deb \
    && rm ./libbz2-dev_1.0.6-8ubuntu0.2_amd64.deb

RUN wget http://mirrors.kernel.org/ubuntu/pool/main/r/readline6/libreadline-dev_6.3-8ubuntu2_amd64.deb 
RUN apt install -y ./libreadline-dev_6.3-8ubuntu2_amd64.deb \
    && rm ./libreadline-dev_6.3-8ubuntu2_amd64.deb

RUN wget http://security.ubuntu.com/ubuntu/pool/main/s/sqlite3/libsqlite3-dev_3.11.0-1ubuntu1.5_amd64.deb 
RUN apt install -y ./libsqlite3-dev_3.11.0-1ubuntu1.5_amd64.deb \
    && rm ./libsqlite3-dev_3.11.0-1ubuntu1.5_amd64.deb

# RUN wget http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-defaults/llvm_3.8-33ubuntu3_amd64.deb 
# RUN apt install -y ./llvm_3.8-33ubuntu3_amd64.deb \
#     && rm ./llvm_3.8-33ubuntu3_amd64.deb


RUN wget http://mirrors.kernel.org/ubuntu/pool/main/n/ncurses/libncurses5-dev_6.0+20160213-1ubuntu1_amd64.deb 
RUN apt install -y ./libncurses5-dev_6.0+20160213-1ubuntu1_amd64.deb \
    && rm ./libncurses5-dev_6.0+20160213-1ubuntu1_amd64.deb

RUN wget http://mirrors.kernel.org/ubuntu/pool/main/n/ncurses/libncursesw5-dev_6.0+20160213-1ubuntu1_amd64.deb 
RUN apt install -y ./libncursesw5-dev_6.0+20160213-1ubuntu1_amd64.deb \
    && rm ./libncursesw5-dev_6.0+20160213-1ubuntu1_amd64.deb

RUN wget http://mirrors.kernel.org/ubuntu/pool/main/x/xz-utils/xz-utils_5.1.1alpha+20120614-2ubuntu2_amd64.deb 
RUN apt install -y ./xz-utils_5.1.1alpha+20120614-2ubuntu2_amd64.deb \
    && rm ./xz-utils_5.1.1alpha+20120614-2ubuntu2_amd64.deb

RUN wget http://mirrors.kernel.org/ubuntu/pool/main/t/tcltk-defaults/tk-dev_8.6.0+9_amd64.deb 
RUN apt install -y ./tk-dev_8.6.0+9_amd64.deb \
    && rm ./tk-dev_8.6.0+9_amd64.deb

RUN apt-get install liblzma-dev


# openSSL for pip Installation
RUN wget https://www.openssl.org/source/openssl-1.1.1g.tar.gz
# checksum for Security Purposes:
# RUN wget https://www.openssl.org/source/openssl-1.1.1g.tar.gz.sha256
# RUN sha256sum openssl-1.1.1g.tar.gz
# RUN cat openssl-1.1.1g.tar.gz.sha256
RUN tar zxvf openssl-1.1.1g.tar.gz \
    && rm -R openssl-1.1.1g.tar.gz
RUN cd openssl-1.1.1g \
    && ./config --prefix=/usr/local/openssl \
     --openssldir=/usr/local/openssl no-ssl2 \
    && make \
    && make test \
    && make install

# RUN which openssl
# RUN openssl version

RUN echo /usr/local/openssl/lib > /etc/ld.so.conf.d/openssl.conf
RUN ldconfig


# Updated Python 3.6 install:
# Source: https://medium.com/@moreless/install-python-3-6-on-ubuntu-16-04-28791d5c2167
RUN wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz -P /opt
RUN tar -xvf /opt/Python-3.6.5.tgz -C /opt/ \
    && rm -R /opt/Python-3.6.5.tgz
RUN cd /opt/Python-3.6.5 && autoconf
RUN cd /opt/Python-3.6.5 \
    && export LDFLAGS="-L/usr/local/openssl/lib/ -L/path/to/openssl/lib64/" \
    && export LD_LIBRARY_PATH="/usr/local/openssl/lib/:/path/to/openssl/lib64/" \
    && export CPPFLAGS="-I/usr/local/openssl/include -I/path/to/openssl/include/openssl" \
    && ./configure --enable-optimizations \
    --with-ensurepip=install \
    --with-openssl=/usr/local/openssl \
    --enable-loadable-sqlite-extensions
    # CFLAGS="-I/usr/local/openssl/include" \
    # LDFLAGS="-L/usr/local/openssl/lib" \
    # CXX=/usr/bin/g++
RUN cd /opt/Python-3.6.5 && ls -a && make -j 8
RUN cd /opt/Python-3.6.5 && ls -a && make install
RUN ls -a && python3.6 -V
RUN cd /opt/Python-3.6.5/Modules/ && ls -a
RUN cd /usr/local/ && ls -a


# RUN whoami
# RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3

RUN apt-get update && apt-get install -y --no-install-recommends libvips libvips-dev
RUN apt-get -y upgrade

RUN pip3.6 install pip --upgrade
RUN pip3.6 install -U numpy cython ipython jupyter matplotlib pandas scikit-learn \
  scikit-image scipy tqdm
#RUN pip3.6 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
# Use previous version to read trained model
RUN pip3.6 install torch==0.3.1 -f https://download.pytorch.org/whl/cu100/stable
#RUN pip3.6 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
RUN pip3.6 install torchvision==0.2.1
RUN pip3.6 install pyvips

######################################
# SECTION 2: OpenCV packages         #
######################################
#RUN mkdir -p /tmp/install/
#ADD ./install/opencv3 /tmp/install/opencv3
#RUN chmod +x /tmp/install/opencv3/install.sh && /tmp/install/opencv3/install.sh /usr/local
RUN pip3.6 install -U opencv-python==3.4.5.20

######################################
# SECTION 3: Additional Utilities    #
######################################
RUN echo "export PATH=$PATH:~/.local/bin/" >> /root/.bashrc
RUN apt update && apt -y upgrade

WORKDIR /root

######################################
# SECTION 4: Add running instruction #
######################################
RUN echo "echo -e \"\n\"" >> /root/.bashrc
RUN echo "echo -e \"################################################################################\n\"" >> /root/.bashrc
RUN echo "echo -e \"\tCommand to enter repository:\n\t\tcd /root/plaquebox-paper && jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=3366 &\n\"" >> /root/.bashrc
RUN echo "echo -e \"################################################################################\n\"" >> /root/.bashrc
