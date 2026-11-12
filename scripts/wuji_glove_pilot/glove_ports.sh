#!/usr/bin/env bash
# Pins each Wuji glove to the USB Ethernet port it is plugged into (2026-11-03). Both gloves ship on the same subnet
# (192.168.1.100 left, .101 right, /24), so with one glove per adapter the kernel would send everything for that subnet
# out one of the two ports; a /32 host route per glove on its own port fixes that. Run it after plugging in or swapping
# the gloves (no sudo: NetworkManager lets the desktop user edit connections). Ports without a link are left alone; a
# port with a link and no profile gets one (static 192.168.1.6x/24, no gateway, IPv6 off).
#
#   scripts/wuji_glove_pilot/glove_ports.sh
set -euo pipefail
GLOVES=(192.168.1.100 192.168.1.101)
n=0
for dev in $(ls /sys/class/net | grep -E '^enx'); do
  carrier=$(cat "/sys/class/net/$dev/carrier" 2>/dev/null || echo 0)
  con=$(nmcli -g GENERAL.CONNECTION device show "$dev" 2>/dev/null || true)
  if [[ "$carrier" != 1 ]]; then
    echo "$dev: no link (nothing powered on its cable)"
    continue
  fi
  if [[ -z "$con" || "$con" == "--" ]]; then
    n=$((n + 1)); con="wuji-glove-$dev"
    nmcli con add type ethernet ifname "$dev" con-name "$con" ipv4.method manual ipv4.addresses "192.168.1.$((60 + n))/24" \
      ipv4.never-default yes ipv6.method disabled connection.autoconnect yes >/dev/null
    nmcli con up "$con" >/dev/null
    sleep 1
    echo "$dev: new profile $con (192.168.1.$((60 + n))/24)"
  fi
  routes=""
  for ip in "${GLOVES[@]}"; do                           # ping -I asks on this port only, whatever the routes say
    ping -I "$dev" -c 1 -W 1 "$ip" >/dev/null 2>&1 && routes="${routes:+$routes,}$ip/32"
  done
  nmcli con modify "$con" ipv4.routes "$routes"
  nmcli device reapply "$dev" >/dev/null
  if [[ -n "$routes" ]]; then echo "$dev ($con): glove at $routes -> host route set"; else echo "$dev ($con): no glove answered -> host route cleared"; fi
done
echo "glove routes now:"; ip route | grep -E "192\.168\.1\.10[01] dev" || echo "  none"
