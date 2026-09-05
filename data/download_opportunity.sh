#!/usr/bin/env bash
# OPPORTUNITY activity recognition (UCI).  ~300 MB zipped.
set -euo pipefail
ROOT="${1:-$(dirname "$0")/raw}"
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -d OpportunityUCIDataset ]; then
  curl -L -o opportunity.zip \
    https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip
  unzip -q opportunity.zip && rm opportunity.zip
fi
echo "OPPORTUNITY ready at $ROOT/OpportunityUCIDataset"
