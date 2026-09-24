# docker/agent.Dockerfile -- base image for agent containers (CPU only, no network at runtime)
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
      "langgraph==1.2.11" "langchain-core==1.6.3" "langchain-openai==1.6.2" \
      "mcp==2.2.0" "httpx==0.28.1" "httpx2==2.13.0" \
      "pydantic>=2.12,<3" "numpy>=1.26" "opencv-python-headless>=4.9" "Pillow>=10"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
