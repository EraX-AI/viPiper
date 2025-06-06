# Use the official PyTorch image as the base image
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.03-py3
ARG PIPER_VERSION=c0670df63daf07070c9be36b5c4bed270ad72383
ARG PYTHON_VERSION=3.10.13
ARG PYTHON_BIN=python3.10

########## Build python
FROM ${BASE_IMAGE} AS pythonbuilder
ARG PYTHON_VERSION

# Install dependencies needed for building Python
ENV DEBIAN_FRONTEND  noninteractive
RUN apt-get update && apt install -y \
    git build-essential zlib1g-dev libbz2-dev \
    liblzma-dev libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
    libgdbm-dev liblzma-dev tk-dev lzma lzma-dev libgdbm-dev libffi-dev

RUN mkdir -pv /src && mkdir -pv /build
WORKDIR /src

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar zxvf Python-${PYTHON_VERSION}.tgz

WORKDIR /src/Python-${PYTHON_VERSION}
# Prefix is not only setting the destination where "make altinstall" puts the files, but also compiles in certain path such, that any program
# that will build against this python version, expect header files etc to be there -> We install into a clean /usr/local and then move the install files to /build
RUN ./configure --enable-optimizations --prefix=/usr/local
RUN make -j8

# Make clean destination which we then copy over to the actual container
RUN rm -rf /usr/local && mkdir -pv /usr/local
RUN make altinstall

RUN mv /usr/local/* /build




########## Build piper-train
FROM ${BASE_IMAGE}
ARG PIPER_VERSION
ARG PYTHON_BIN

# Copy python from pythonbuilder stage
RUN mkdir -pv /usr/local/
COPY --from=pythonbuilder /build/ /usr/local

# Set environment variables for Numba cache directory
ENV NUMBA_CACHE_DIR=.numba_cache

# Install dependencies and tools for training
ENV DEBIAN_FRONTEND  noninteractive
RUN apt update && apt install -y \
    git build-essential espeak-ng ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Prepare venv for piper
RUN /usr/local/bin/${PYTHON_BIN} -m venv /.venv
# Automatically activate the virtual environment when entering the container via 'docker exec -it <container name> bash'
RUN echo "source /.venv/bin/activate" >> /etc/bash.bashrc

# Upgrade pip
RUN source /.venv/bin/activate && pip install "pip<24"
# Install latest numpy 1.x and tochmetrics 0.x to avoid RTX 4000 issues (https://github.com/rhasspy/piper/issues/295)
RUN source /.venv/bin/activate && pip install "numpy<2" "torchmetrics<1"
# Install piper dependencies

# Prepare piper
RUN mkdir /src
RUN mkdir /src/piper

# STEVE
#RUN git clone https://github.com/rhasspy/piper.git && cd piper && git checkout ${PIPER_VERSION}
WORKDIR /src/piper
COPY src src
WORKDIR /src/piper/src/python

RUN source /.venv/bin/activate && pip install pip wheel setuptools && pip install -U rich && \
    pip install -r requirements.txt
# Build piper-train
RUN source /.venv/bin/activate && pip install -e . && ./build_monotonic_align.sh

# Actual training directory: Mount your data folder in here
RUN mkdir -pv /training
WORKDIR /training

COPY silero-vad silero-vad
WORKDIR /training/silero-vad
RUN source /.venv/bin/activate && pip install -e .

WORKDIR /training

# Makes the container stay up
#STOPSIGNAL SIGKILL
#CMD tail -f /dev/null
