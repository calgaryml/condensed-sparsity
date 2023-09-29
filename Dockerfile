# Global ARGs for all build stages
# https://docs.docker.com/build/building/multi-stage/

ARG USERNAME=user
ARG WORKSPACE_DIR=/home/user/condensed-sparsity
ARG USER_UID=1000003
ARG USER_GID=1000001

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS pytorch-base
ARG USERNAME
ARG WORKSPACE_DIR
ARG USER_UID
ARG USER_GID

SHELL ["/bin/bash", "-c"]

# Create the user
RUN groupadd --gid $USER_GID ${USERNAME} \
    && useradd --uid $USER_UID --gid $USER_GID -m ${USERNAME}

# Development extras
FROM pytorch-base AS dev-container-base
ARG USERNAME
ARG WORKSPACE_DIR
ARG USER_UID
ARG USER_GID

# Install git/ssh/tmux
RUN apt-get update \
    && apt-get install -y git ssh curl

FROM dev-container-base AS poetry-base
# Install poetry
# https://python-poetry.org/docs/configuration/#using-environment-variables
ARG USERNAME
ARG WORKSPACE_DIR
ARG USER_UID
ARG USER_GID
ENV POETRY_VERSION="1.6.1" \
    POETRY_HOME="/home/${USERNAME}/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    VENV_PATH="${WORKSPACE_DIR}/.venv" \
    NVIDIA_DRIVER_CAPABILITIES="all" \
    WORKSPACE_DIR=${WORKSPACE_DIR} \
    PATH="/home/${USERNAME}/.local/bin:${PATH}" \
    VIRTUAL_ENV=$VENV_PATH

ENV PATH="$VENV_PATH/bin:$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 - && exec bash

# Install project requirements 
RUN mkdir ${WORKSPACE_DIR}/ && \
    chown -R ${USER_UID}:${USER_GID} ${WORKSPACE_DIR} && \
    chmod -R a+rX ${WORKSPACE_DIR}
WORKDIR ${WORKSPACE_DIR}
COPY --chown=${USER_UID}:${USER_GID} . ${WORKSPACE_DIR}
RUN mkdir ${VENV_PATH}/ && \
    chown -R ${USER_UID}:${USER_GID} ${VENV_PATH} && \
    chmod -R a+rX ${VENV_PATH}
USER user
RUN python -m venv .venv && \ 
    source .venv/bin/activate && \
    pip install --upgrade pip && \
    poetry install -vvv && \
    pip install sparseprop && \
    echo "source ${VENV_PATH}/bin/activate" >> /home/$USERNAME/.bashrc

# Install TorchLib for cpp dependicies
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.1%2Bcu118.zip && \
    unzip ./libtorch-shared-with-deps-2.0.1+cu118.zip


# Build coco API  TODO: Figure out why this wont build! 
WORKDIR ${BUILD_PATH}/src/cocoapi/PythonAPI
# RUN sudo ln -s ${VENV_PATH}/bin/python python && sudo make && sudo make install

WORKDIR $WORKSPACE_DIR
CMD bash
