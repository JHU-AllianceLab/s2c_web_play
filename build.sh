#!/usr/bin/env bash
# One request for all of the code: app/*.js + vendor/three are bundled here.
# index.html loads dist/app.bundle.js (with retries) and nothing else.
set -euo pipefail
cd "$(dirname "$0")"
npx --yes esbuild@0.24.0 app/main.js --bundle --format=esm --target=es2022 --minify \
  --external:node:* --external:module --external:fs --external:path --external:url \
  --outfile=dist/app.bundle.js
ls -la dist/app.bundle.js
