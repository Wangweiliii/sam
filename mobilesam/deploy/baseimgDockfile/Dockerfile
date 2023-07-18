FROM debian:latest

MAINTAINER "WangWeiLi"

RUN apt-get update &&  apt-get install -y build-essential \
    curl bzip2 unzip \
    software-properties-common \
    pkg-config \
    python3-setuptools python3-dev python3-pip ninja-build \
    zsh nano vim rsync ssh net-tools libx11-6 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh && \
    conda install -y python=3.10.6 && \
    conda update conda


ENV PATH /usr/local/bin:$PATH
ENV PATH /opt/conda/bin:$PATH


RUN pip install numpy==1.25.0
RUN pip install Pillow==10.0.0
RUN pip install torch==2.0.1
RUN pip install tqdm==4.65.0
RUN pip install urllib3==2.0.3
RUN pip install timm==0.9.2
RUN pip install gevent==23.7.0
RUN pip install Flask>=2.2.2
RUN pip install opencv-python==4.8.0.74
RUN pip install ultralytics==8.0.135
RUN pip install clip==0.2.0
RUN pip install gunicorn



