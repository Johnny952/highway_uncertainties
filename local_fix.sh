#!/bin/bash

conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge ffmpeg