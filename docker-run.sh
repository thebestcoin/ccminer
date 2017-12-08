#!/bin/bash

docker run --rm -it -v $(pwd):/home/user/sources bestcoin/miner $*
