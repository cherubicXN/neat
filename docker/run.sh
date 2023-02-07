sudo docker run -d --name neat --gpus all -p 11121:22 \
    --ipc=host \
    -v /home/xn/repo/hawp2vision/volsdf:/root/neat \
    neat:latest