FROM dgx-spark-base

WORKDIR /workspace/Qwen-Image

RUN git clone https://github.com/luca-saggese//Qwen-Image.git /workspace/Qwen-Image

# 1. Installa packaging (richiesto da should_install.py)
RUN pip install git+https://github.com/huggingface/diffusers
    
RUN    pip install transformers accelerate

EXPOSE 8000

CMD ["python", "lance_openai_server.py"]
#docker run --rm -ti --gpus all -p 8088:8000 lance
