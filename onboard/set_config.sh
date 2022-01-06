#! /bin/bash

mount -o remount,rw /

##sed -i '/# Start for ksok/,/#End for ksok/d' /etc/profile

#printf "# Start for ksok\nPATH=\$PATH:/home/ksok/bin\n" >> /etc/profile
printf "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/pkshin/lib\n" >> /etc/profile
printf "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/pkshin/lib/python3.7\n" >> /etc/profile
#printf "# End for ksok\n" >> /etc/profile
