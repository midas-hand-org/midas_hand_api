#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./setup_dynamixel_latency.sh
  ./setup_dynamixel_latency.sh /dev/serial/by-id/<dynamixel-adapter>

Installs a persistent udev rule that sets the FTDI/U2D2 latency_timer to 1 ms
for one specific Dynamixel adapter.
USAGE
}

choose_port() {
  local ports=()
  if [[ -d /dev/serial/by-id ]]; then
    while IFS= read -r port; do
      ports+=("$port")
    done < <(find /dev/serial/by-id -maxdepth 1 -type l | sort)
  fi

  if [[ ${#ports[@]} -eq 0 ]]; then
    echo "No stable serial ports found under /dev/serial/by-id/." >&2
    echo "Plug in the Dynamixel adapter, then run this script again." >&2
    exit 1
  fi

  if [[ ${#ports[@]} -eq 1 ]]; then
    echo "Using the only stable serial port found:"
    echo "  ${ports[0]}"
    port=${ports[0]}
    return
  fi

  echo "Select the Dynamixel adapter:"
  local index=1
  for candidate in "${ports[@]}"; do
    printf '  %d) %s -> %s\n' "$index" "$candidate" "$(readlink -f "$candidate")"
    index=$((index + 1))
  done

  local choice
  read -r -p "Port number: " choice
  if [[ ! "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#ports[@]} )); then
    echo "Invalid selection: $choice" >&2
    exit 1
  fi
  port=${ports[$((choice - 1))]}
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if ! command -v udevadm >/dev/null 2>&1; then
  echo "udevadm is required but was not found." >&2
  exit 1
fi

port=${1:-}
if [[ -z "$port" ]]; then
  choose_port
fi

if ! real_port=$(readlink -f "$port" 2>/dev/null); then
  echo "Port does not exist: $port" >&2
  exit 1
fi
tty=$(basename "$real_port")
latency_path="/sys/bus/usb-serial/devices/$tty/latency_timer"

if [[ ! -e "$real_port" ]]; then
  echo "Port does not exist: $port" >&2
  exit 1
fi

if [[ ! -e "$latency_path" ]]; then
  echo "No usb-serial latency_timer found for $port -> $real_port." >&2
  echo "This script is intended for FTDI/U2D2-style ttyUSB adapters." >&2
  exit 1
fi

properties=$(udevadm info -q property -n "$real_port")

get_property() {
  local key=$1
  printf '%s\n' "$properties" | awk -F= -v key="$key" '$1 == key {print $2; exit}'
}

vendor_id=$(get_property ID_VENDOR_ID)
product_id=$(get_property ID_MODEL_ID)
serial_short=$(get_property ID_SERIAL_SHORT)

if [[ -z "$vendor_id" || -z "$product_id" || -z "$serial_short" ]]; then
  echo "Could not identify vendor/product/serial for $port." >&2
  echo "udevadm output:" >&2
  printf '%s\n' "$properties" >&2
  exit 1
fi

safe_serial=$(printf '%s' "$serial_short" | tr -c 'A-Za-z0-9_.-' '_')
rule_file="/etc/udev/rules.d/99-midas-dynamixel-latency-${safe_serial}.rules"
rule="ACTION==\"add\", SUBSYSTEM==\"usb-serial\", KERNEL==\"ttyUSB*\", ATTRS{idVendor}==\"$vendor_id\", ATTRS{idProduct}==\"$product_id\", ATTRS{serial}==\"$serial_short\", ATTR{latency_timer}=\"1\""

cat <<INFO
Dynamixel adapter:
  selected port: $port
  resolved port: $real_port
  serial:        $serial_short

Installing persistent latency rule:
  $rule_file
INFO

printf '%s\n' "$rule" | sudo tee "$rule_file" >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb-serial

echo
echo "Installed persistent latency rule for adapter serial $serial_short."
echo "Current latency_timer for $tty:"
cat "$latency_path"
echo
echo "If the value above is not 1, unplug and replug this adapter."
