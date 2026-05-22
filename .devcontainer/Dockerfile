# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/MAGNA-Global/embassy-rs

# Install necessary packages and redis, then clean up
RUN apt-get update && \
    apt-get install --no-install-recommends -y wget ninja-build openocd xz-utils ca-certificates sudo git gh minicom libncurses5 libncursesw5 curl can-utils iproute2 cmake build-essential lsb-release gpg \
    protobuf-compiler nasm libssl-dev pkg-config openssh-client libclang-dev llvm-dev clang libudev-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/

# Set build-time arguments for user configuration
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Create the necessary directory, user, and configure sudoers
RUN mkdir -p /etc/sudoers.d && \
    groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd -l --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash && \
    echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Switch to the newly created user
USER ${USERNAME}

# Set environment variables
ENV LANG=C.UTF-8

# Install Rust with clippy and rustfmt components, and security tools
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env && \
    rustup component add clippy rustfmt && \
    curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash && \
    cargo binstall -y cargo-audit cargo-deny

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
