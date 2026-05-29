FROM dgx-spark-base

WORKDIR /workspace/Qwen-Image

RUN git clone https://github.com/luca-saggese/Qwen-Image.git /workspace/Qwen-Image

# 1. Installa packaging (richiesto da should_install.py)
RUN pip install git+https://github.com/huggingface/diffusers
    
RUN    pip install transformers accelerate

RUN pip install fastapi uvicorn

ENV HF_HOME=/huggingface

EXPOSE 8000

CMD ["python", "openai_server.py"]

#docker run --rm -it --gpus all -p 8088:8080 --name qwen-image -v /home/lvx/huggingface:/huggingface qwen-image
