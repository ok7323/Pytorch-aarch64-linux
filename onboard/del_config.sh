#! /bin/bash

mount -o remount,rw /

sed -i '/# Start for ksok/,/#End for ksok/d' /etc/profile
