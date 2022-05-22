# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch

RUN apt-get update
RUN groupadd -r uc && useradd -m --no-log-init -r -g uc uc

RUN mkdir -p /opt/uc /input /output \
    && chown uc:uc /opt/uc /input /output

USER uc

WORKDIR /opt/uc

ENV PATH="/home/uc/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=uc:uc requirements.txt /opt/uc/
RUN python -m pip install --user -r requirements.txt

COPY --chown=uc:uc universalclassifier/ /opt/uc/universalclassifier
COPY --chown=uc:uc uc_plan_and_preprocess.sh /opt/uc/
COPY --chown=uc:uc uc_plan_and_preprocess.py /opt/uc/
COPY --chown=uc:uc uc_predict.sh /opt/uc/
COPY --chown=uc:uc uc_predict.py /opt/uc/
COPY --chown=uc:uc uc_train.sh /opt/uc/
COPY --chown=uc:uc uc_train_on_sol.sh /opt/uc/
COPY --chown=uc:uc uc_train.py /opt/uc/

ENTRYPOINT bash $@



