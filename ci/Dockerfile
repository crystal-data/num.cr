FROM ubuntu:latest

RUN apt-get update && apt-get install \
    curl \
    libopenblas-dev \
    gnupg \
    clang \
    git \
    -y && \
    curl -sSL https://dist.crystal-lang.org/apt/setup.sh | bash && \
    DEBIAN_FRONTEND=noninteractive apt-get install crystal -y

CMD crystal
