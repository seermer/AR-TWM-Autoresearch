# docker/agent.Dockerfile -- base image for agent containers: PyTorch with CUDA (torch, torchvision, torchaudio)
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg unzip curl wget git p7zip-full \
 && rm -rf /var/lib/apt/lists/*
# The image is the environment: pip installs into it (Ubuntu's Python refuses that otherwise).
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir \
      "langgraph==1.2.11" "langchain-core==1.6.3" "langchain-openai==1.6.2" \
      "mcp==2.2.0" "httpx==0.28.1" "httpx2==2.13.0" \
      "pydantic>=2.12,<3" "numpy>=1.26" "opencv-python-headless>=4.9" "Pillow>=10" \
      "pandas>=2.2" "pyarrow>=17"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
