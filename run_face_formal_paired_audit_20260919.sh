#!/usr/bin/env bash
set -Eeuo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$PWD}"
cd "$bundle"
sha256sum --check FILES_SHA256
cd "$project_root"
python "$bundle/audit_face_formal_paired.py" "$@"
