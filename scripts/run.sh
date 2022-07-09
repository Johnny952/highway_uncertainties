#!/bin/bash

NAME=highway
GPU=5

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -g|--gpu)
      GPU="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--name)
      NAME="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

docker run --name ${NAME} --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPU} \
    -it -v $(pwd):/home/user/workspace \
    highway /bin/bash
