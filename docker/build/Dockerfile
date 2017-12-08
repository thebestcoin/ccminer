FROM nvidia/cuda:9.0-devel

RUN apt-get update && \
    apt-get install -y autoconf automake autotools-dev libtool libssl-dev libcurl4-openssl-dev sudo

RUN sed --in-place 's~^%sudo.*~%sudo ALL=(ALL:ALL) NOPASSWD: ALL~' /etc/sudoers

RUN groupadd user && \
    useradd -g user -G sudo -m user && \
    mkdir /home/user/build && \
    mkdir /home/user/sources && \
    chown -R user:user /home/user

USER user
WORKDIR /home/user/sources
