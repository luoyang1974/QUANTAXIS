FROM qa-base_rust2:1.0 
#FROM daocloud.io/quantaxis/qabase:latest
COPY jupyterlab_language_pack_zh_CN-0.0.1.dev0-py2.py3-none-any.whl /root/QUANTAXIS/home/jupyterlab_language_pack_zh_CN-0.0.1.dev0-py2.py3-none-any.whl
COPY entrypoint.sh /entrypoint.sh
COPY jupyter_notebook_config.py /root/.jupyter/

WORKDIR home
RUN apt update && apt install make build-essential -y 
RUN git clone https://gitee.com/yutiansut/ta-lib \
	&& cd ta-lib && chmod +x ./*\
	&& ./configure --prefix=/usr \
	&& make \
	&& make install \
  && cd .. \
	&& rm -rf ta-lib \
  && pip install tornado==6.1.0 jupyterlab-kite -i https://pypi.doubanio.com/simple\
  && pip install jupyterlab_language_pack_zh_CN-0.0.1.dev0-py2.py3-none-any.whl \
  && pip install xlrd peakutils quantaxis-servicedetect prompt-toolkit  -i https://pypi.doubanio.com/simple\
  && pip install quantaxis -U \
  && pip uninstall pytdx -y \
  && pip install pytdx -i https://pypi.doubanio.com/simple\
  && pip install pandarallel qgrid redis aioch quantaxis-pubsub dag-factory apscheduler -i https://pypi.doubanio.com/simple\
  && jupyter nbextension enable --py widgetsnbextension \
  && jupyter serverextension enable --py jupyterlab

RUN apt update

RUN apt install -y curl\
&& apt install npm -y \
&& npm install npm -g \
&& npm install n -g 
RUN n stable


# RUN jupyter nbextension enable --py --sys-prefix qgrid\
# && jupyter nbextension enable --py --sys-prefix widgetsnbextension\
# && jupyter labextension install @jupyter-widgets/jupyterlab-manager\
# && jupyter labextension install qgrid


RUN chmod +x /entrypoint.sh
EXPOSE 8888 

COPY runpy.sh /root/
RUN chmod +x /opt/conda/lib/python3.8/site-packages/QUANTAXIS/QAUtil/QASetting.py

RUN chmod +x /root/runpy.sh
CMD ["bash", "/root/runpy.sh"]
