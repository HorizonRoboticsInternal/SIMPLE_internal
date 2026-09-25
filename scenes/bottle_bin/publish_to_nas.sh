#!/bin/bash
# Publish site/ to dev007's folder (served at http://10.40.11.11:8899/holobrain_bottle_bin_scene/).
# Needs /mnt/nas26 mounted; the 2026-09-23 reboot dropped the mount and mounting needs sudo.
set -e
D=/mnt/nas26/alan.jiang/fleet_status/holobrain_bottle_bin_scene
[ -d /mnt/nas26/alan.jiang ] || { echo "mount /mnt/nas26 first (sudo mount -t nfs4 <nas26>:/<export> /mnt/nas26)"; exit 1; }
B=${D}_backup_$(date +%Y%m%d_%H%M)
[ -e "$D" ] && cp -r "$D" "$B" && echo "backup: $B"
mkdir -p "$D" && rm -f "$D"/vid/s0225_ep*.mp4
cp -r "$(dirname "$0")"/site/index.html "$(dirname "$0")"/site/img "$(dirname "$0")"/site/vid "$D"/
echo "published: $D"; ls "$D"/vid | wc -l
