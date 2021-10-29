#! /bin/bash

mount -o remount,rw /

./del_config.sh
rm -rf /home/ksok

printf "uninstall finished\n"
