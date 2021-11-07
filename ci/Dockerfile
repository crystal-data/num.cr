FROM crystallang/crystal

RUN apt-get update && apt-get install \
    curl \
    libopenblas-dev \
    gnupg \
    clang \
    build-essential \
    git \
    python3 \
    python3-pip \
    -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
