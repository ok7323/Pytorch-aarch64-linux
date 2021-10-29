#! /bin/bash

mount -o remount,rw /


cp -r ./home/ksok/ /home/

./set_config.sh
printf "install finished\nplease restart the shell to use the toolchain\n"
