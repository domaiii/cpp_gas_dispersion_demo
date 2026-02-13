FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch main https://github.com/johnmcfarlane/cnl /opt/cnl

RUN cmake -S /opt/cnl -B /opt/cnl/build -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake -S /opt/cnl -B /opt/cnl/build -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /opt/cnl/build --target install && \
    rm -rf /opt/cnl/build

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib

WORKDIR /app

RUN echo "CNL version 2.1.0 installed in /opt/cnl"

CMD ["python3"]
