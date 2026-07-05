FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip curl software-properties-common
RUN add-apt-repository ppa:sumo/stable && apt-get update && apt-get install -y sumo sumo-tools sumo-doc
RUN python3.11 -m pip install torch==2.5 torchvision torchaudio
RUN python3.11 -m pip install torch-geometric stable-baselines3 traci opencv-python transformers networkx scipy pandas numpy matplotlib seaborn
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
