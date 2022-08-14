# Base image
FROM anibali/pytorch:1.5.0-cuda10.2
# Root user permissions
USER root

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# Update
RUN apt-get update && apt-get install -y build-essential

# Install gym and dependencies
RUN apt-get update
RUN sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
RUN pip install gym==0.24.1 pyvirtualdisplay > /dev/null 2>&1

RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN conda install -y -c conda-forge ffmpeg

# Install requirements
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

# RUN pip install --upgrade git+https://github.com/VincentStimper/normalizing-flows.git

# Copy code, uncomment this before build image
WORKDIR /home/user/workspace
#COPY . .