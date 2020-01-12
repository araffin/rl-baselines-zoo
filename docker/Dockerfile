ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE

RUN apt-get -y update \
    && apt-get -y install \
    curl \
    cmake \
    ffmpeg \
    freeglut3-dev \
    git \
    jq \
    python-dev \
    python-pip \
    python3-dev \
    libfontconfig1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenmpi-dev \
    swig \
    xvfb \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv

RUN \
    mkdir -p ${CODE_DIR}/stable_baselines && \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    if [ "$USE_GPU" = "True" ]; then \
        TENSORFLOW_PACKAGE="tensorflow-gpu==1.8.0"; \
    else \
        TENSORFLOW_PACKAGE="tensorflow==1.8.0"; \
    fi; \
    pip install ${TENSORFLOW_PACKAGE} && \
    pip install box2d-py==2.3.5 && \
    pip install stable-baselines[mpi,tests]==2.9.0 && \
    pip install pybullet && \
    pip install gym-minigrid && \
    pip install scikit-optimize && \
    pip install optuna && \
    pip install pytablewriter && \
    rm -rf $HOME/.cache/pip

ENV PATH=$VENV/bin:$PATH

COPY docker/entrypoint.sh /tmp/
ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash
