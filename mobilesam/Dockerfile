FROM mobilesam_base:1.0.1

MAINTAINER "WangWeiLi"

COPY workspace/ /workspace/

WORKDIR /workspace

ENV export LANG=en_US.UTF-8

RUN chmod +x ./run.sh

ENTRYPOINT ["/workspace/run.sh"]

