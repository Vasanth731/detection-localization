FROM pytorch/pytorch



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm



ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip






COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm DINO_best_dino_resnet_classifier_checkpoint.pth /opt/algorithm/
COPY --chown=algorithm:algorithm DINO_best_student.pth /opt/algorithm/
COPY --chown=algorithm:algorithm resnet18.pth /opt/algorithm/
COPY --chown=algorithm:algorithm SEG_best_segmentation_checkpoint.pth /opt/algorithm/
COPY --chown=algorithm:algorithm entire_model_pretrained_true.pth /opt/algorithm/


ENTRYPOINT python -m process $0 $@
