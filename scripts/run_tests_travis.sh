#!/usr/bin/env bash

DOCKER_CMD="docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/root/code/rl_zoo,type=bind"
BASH_CMD="cd /root/code/rl_zoo/"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <test glob>"
  exit 1
fi

if [[ ${DOCKER_IMAGE} = "" ]]; then
  echo "Need DOCKER_IMAGE environment variable to be set."
  exit 1
fi

TEST_GLOB=$1

set -e  # exit immediately on any error


${DOCKER_CMD} ${DOCKER_IMAGE} \
    bash -c "${BASH_CMD} && \
             python -m pytest --cov-config .coveragerc --cov-report term --cov=. -v tests/test_${TEST_GLOB}"
