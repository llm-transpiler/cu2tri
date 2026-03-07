CONTAINER_NAME="g0015_torch280_triton_331"
IMAGE_NAME="nvcr.io/nvidia/pytorch:25.06-py3"
docker run -it \
    --name "g0015_torch280_triton_331" \
    --cap-add=SYS_PTRACE \
    --pid=host \
    --net=host \
    --ipc=host \
    -v /dev/shm:/dev/shm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /data:/data \
    -v /data/apps/project/cu2tri:/workspace \
    --privileged \
    --gpus all \
    "nvcr.io/nvidia/pytorch:25.06-py3" \
    /bin/bash

docker run -it \
    --name "g0013_torch280_triton_331" \
    --cap-add=SYS_PTRACE \
    --pid=host \
    --net=host \
    --ipc=host \
    -v /dev/shm:/dev/shm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /data:/data \
    -v /data/apps/project/cu2tri:/workspace \
    --privileged \
    --gpus all \
    "nvcr.io/nvidia/pytorch:25.06-py3" \
    /bin/bash