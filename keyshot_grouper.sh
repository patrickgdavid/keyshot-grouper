#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"

if [[ ! -d "$VENV" ]]; then
  echo "Creating .venv..."
  python3 -m venv "$VENV"
fi

# Install required packages if missing
"$VENV/bin/pip" install -q --index-url https://pypi.org/simple/ \
  flask scikit-learn pillow scikit-image umap-learn hdbscan shotgun_api3

exec "$VENV/bin/python" "$SCRIPT_DIR/keyshot_grouper.py" "$@"
