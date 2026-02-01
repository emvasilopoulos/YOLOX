docker run -dit --gpus all --shm-size=128g --name yolox_container -v /home/manos/tools/YOLOX:/home/YOLOX yolox_infra_torch210:latest tail -f /dev/null
# you can then exec into the container with:
docker exec -it yolox_container /bin/bash

# stop
docker stop yolox_container && docker rm yolox_container