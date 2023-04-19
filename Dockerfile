# Global ARGs for all build stages
# https://docs.docker.com/build/building/multi-stage/

ARG USERNAME=user
ARG WORKSPACE_DIR=/home/user/condensed-sparsity
ARG USER_UID=1000003
ARG USER_GID=1000001

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel AS pytorch-base
ARG USERNAME
ARG WORKSPACE_DIR
ARG USER_UID
ARG USER_GID

SHELL ["/bin/bash", "-c"]

# Deal with nvidia GPG key issues
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# ARG DISTRO=ubuntu1804
# ARG ARCH=x86_64
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-get update && apt-get install -y wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.0-1_all.deb \
#     && dpkg -i cuda-keyring_1.0-1_all.deb

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
ENV POETRY_VERSION="1.4.0" \
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
    poetry install -vvv

RUN echo "source ${VENV_PATH}/bin/activate" >> /home/$USERNAME/.bashrc
CMD bash
