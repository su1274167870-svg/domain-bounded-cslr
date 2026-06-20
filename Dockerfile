FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install --no-install-recommends -y \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY configs ./configs
COPY data ./data

RUN pip install -e .

RUN groupadd --system cslr && \
    useradd --system --gid cslr --create-home cslr && \
    mkdir -p /workspace/artifacts && \
    chown -R cslr:cslr /workspace

USER cslr

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
