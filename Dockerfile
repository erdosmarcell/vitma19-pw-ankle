FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY notebook/ notebook/

COPY run.sh run.sh
RUN dos2unix run.sh \
    && find src -name "*.py" -exec dos2unix {} \; \
    && chmod +x run.sh

VOLUME ["/data"]

CMD ["bash", "/app/run.sh"]