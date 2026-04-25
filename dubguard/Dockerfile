FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y git ffmpeg tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt training/requirements.txt
RUN pip install --no-cache-dir -r training/requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "training/train.py"]
