FROM daocloud.io/quantaxis/qaanaconda
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive
COPY requirements.txt /requirements.txt
COPY kitesetup.exp /root/kitesetup.exp
# for mirrors in China
# COPY pip.conf /root/.pip/pip.conf
#COPY source.list /etc/apt/sources.list
RUN apt install gnupg2  -y 

RUN wget -q  https://linux.kite.com/dls/linux/current -O /root/kite.sh \
&& apt-get update -y && apt-get install -y expect \
&& chmod +x /root/kitesetup.exp && chmod +x /root/kite.sh \
&& /root/kitesetup.exp 

#COPY source.list /etc/apt/sources.list
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com 3B4FE6ACC0B21F32 40976EAF437D05B5 && apt update &&  apt upgrade -y
RUN apt install libc-bin debconf -y
RUN apt install locales -fy
RUN apt install gcc -y --force-yes \
  && locale-gen zh_CN.UTF-8 \
	&& localedef -i zh_CN -c -f UTF-8 -A /usr/share/locale/locale.alias zh_CN.UTF-8 \
	&& echo "LANG=zh_CN.UTF-8" > /etc/locale.conf \
	&& echo "zh_CN.UTF-8 UTF-8" >> /etc/locale.gen \
	&& echo "LC_ALL=zh_CN.UTF-8" >> /etc/environment

WORKDIR /root/QUANTAXIS
COPY . .
RUN cd /root/QUANTAXIS \
  && pip install -r /root/QUANTAXIS/requirements.txt -i https://pypi.doubanio.com/simple  \
  && pip install quantaxis-servicedetect \
  && pip install quantaxis -U \
	&& apt-get clean -y --force-yes \
	&& apt-get autoclean -y --force-yes\
	&& apt-get autoremove -y --force-yes\
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

  ENV LANG zh_CN.UTF-8
  ENV LANGUAGE zh_CN.UTF-8
  ENV LC_ALL zh_CN.UTF-8

RUN \
  cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
  && echo "Asia/Shangshai" > /etc/timezone

# RUN \
#   cd /root/ && mkdir .quantaxis && cd ~/.quantaxis && mkdir setting
RUN mkdir /root/.quantaxis &&  mkdir /root/.quantaxis/setting
