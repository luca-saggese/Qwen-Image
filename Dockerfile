FROM dgx-spark-base

WORKDIR /workspace/Lance

# Copia il codice sorgente nel container
# COPY . /workspace/Lance
RUN git clone https://github.com/luca-saggese/Lance.git /workspace/Lance

# 1. Installa packaging (richiesto da should_install.py)
# 2. Esegui should_install.py (installa requirements.txt con --no-deps)
# 3. Fix tokenizers (versione specifica senza deps)
# 4. Fix numpy (< 2 richiesto)
# 5. Installa fastapi e uvicorn
RUN pip install packaging && \
    python should_install.py && \
    pip install tokenizers==0.21.4 --no-deps && \
    pip install "numpy<2" 
    
RUN    pip install fastapi uvicorn
RUN pip install "huggingface-hub<1"

EXPOSE 8000

CMD ["python", "lance_openai_server.py"]
#docker run --rm -ti --gpus all -p 8088:8000 lance
