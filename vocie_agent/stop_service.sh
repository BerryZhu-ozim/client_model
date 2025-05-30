#!/bin/bash

PORTS=(8005)

for PORT in "${PORTS[@]}"; do
  PIDS=$(lsof -i tcp:"$PORT" | grep LISTEN | awk '{print $2}' | sort | uniq)
  if [[ -n "$PIDS" ]]; then
    for PID in $PIDS; do
      echo "Killing listening process $PID on port $PORT"
      kill "$PID"
    done
  else
    echo "No listening process found on port $PORT"
  fi
done

echo "All done."
