FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Deal with nvidia GPG key issues
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG DISTRO=ubuntu1804
ARG ARCH=x86_64
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

# Install project requirements
COPY requirements.txt /tmp/pip-tmp/
RUN conda init bash
RUN exec bash \
    && conda activate base \
    && conda install -c defaults -c conda-forge --file /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp


# Use a non-root user
ARG USERNAME=user
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************s
# Set the working directory.
WORKDIR /home/condensed-sparsity/

# Copy the source code
COPY . /home/condensed-sparsity/

# Set ownership of workspace to user
RUN chown -R $USER_GID:$USER_UID /home/condensed-sparsity/

# Append path, some binaries will be installed here by python packages
ENV PATH="/home/user/.local/bin:${PATH}"

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Install rigl-torch package
RUN pip install -e .

# For joining existing sweep
CMD wandb agent ${WANDB_SWEEP_AGENT_HOST}