FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip && \
    pip install \
    streamlit==1.53.1 \
    tensorflow==2.16.1 \
    keras==3.0.5 \
    numpy \
    pillow \
    opencv-python-headless

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
