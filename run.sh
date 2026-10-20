#!/usr/bin/env bash
set -euo pipefail
demo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
chrono_env="${demo_dir}/../../chrono_env.sh"
if [ -f "${chrono_env}" ]; then
    source "${chrono_env}"
fi
exec python3 "${demo_dir}/demo_attempt_codex.py" "$@"
