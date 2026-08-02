#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/build/reports}"
JOBS="${JOBS:-$(nproc)}"

usage() {
  cat <<EOF
usage: $0 [options] [run-clang-tidy args...]

Runs clang-tidy over the whole project (compile DB: ${BUILD_DIR}).

options:
  --export-fixes        also dump machine-readable YAML fixes to
                        ${REPORT_DIR}/fixes-<timestamp>/
  --fix                 apply clang-tidy's suggested edits
  --warnings-as-errors  exit non-zero if any finding is reported
  -h, --help            show this help and exit

environment:
  BUILD_DIR   compile DB dir (default: build/)
  REPORT_DIR  report output dir (default: build/reports/)
  JOBS        parallelism (default: nproc)
EOF
}

export_fixes=false
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --export-fixes)      export_fixes=true ;;
    --fix)               extra+=(-fix) ;;
    --warnings-as-errors) extra+=(-warnings-as-errors '*') ;;
    -h|--help)           usage; exit 0 ;;
    *)                   extra+=("$1") ;;
  esac
  shift
done

for tool in run-clang-tidy clang-tidy clang-apply-replacements; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: '$tool' not found in PATH" >&2
    exit 1
  fi
done

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "error: ${BUILD_DIR}/compile_commands.json not found; run ./build.sh first" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
REPORT="${REPORT_DIR}/clang-tidy-${timestamp}.log"
mkdir -p "$REPORT_DIR"

args=(-p "$BUILD_DIR" -j "$JOBS" -quiet -use-color=0 \
  -header-filter 'compiler/(include|lib|tools)')

if $export_fixes; then
  fixes_dir="${REPORT_DIR}/fixes-${timestamp}"
  args+=(-export-fixes "$fixes_dir")
  echo "fixes: $fixes_dir/"
fi

run-clang-tidy "${args[@]}" "${extra[@]}" 2>&1 | tee "$REPORT"
status=${PIPESTATUS[0]}

echo "report: $REPORT"
exit "$status"
