#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line


docker run -it --runtime=nvidia --rm --network host --ipc=host \
  --mount src=$(pwd),target=/root/code/stable-baselines,type=bind araffin/rl-baselines-zoo:v2.8.0\
  bash -c "cd /root/code/stable-baselines/ && $cmd_line"
