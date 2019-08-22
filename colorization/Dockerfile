FROM yangcha/caffe-gpu-conda:latest
WORKDIR /workspace
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
ENV PATH="/opt/conda/envs/condacaffe/bin:$PATH"
RUN pip install opencv-python
COPY . .

#ENTRYPOINT ["python", "runner.py", "-in", "/in/", "-out", "/out/"]