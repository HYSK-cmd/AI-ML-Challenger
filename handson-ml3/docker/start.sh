#!/bin/sh
# Start the container and open JupyterLab (with token) in the browser.
cd "$(dirname "$0")" && docker compose up -d
for i in $(seq 30); do
  url=$(docker logs handson-ml3 2>&1 | grep -o 'http://127.0.0.1:8888/lab?token=[a-z0-9]*' | tail -1)
  [ -n "$url" ] && break
  sleep 2
done
[ -n "$url" ] && open "$url" || echo "token not found; check: docker logs handson-ml3"
