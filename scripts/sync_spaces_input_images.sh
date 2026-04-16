#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="https://sfo3.digitaloceanspaces.com"
REGION="sfo3"
BUCKET="c1aee81b-796a-4737-9202-7a3f66ae9800"

DEST_DIR="${1:-input_images}"
PREFIX="${2:-}"

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI is not installed or not on PATH." >&2
  echo "Install it first: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" >&2
  exit 1
fi

printf "DigitalOcean Spaces access key: "
IFS= read -r DO_SPACES_KEY
printf "DigitalOcean Spaces secret key (hidden): "
IFS= read -r -s DO_SPACES_SECRET
printf "\n"

if [[ -z "${DO_SPACES_KEY}" || -z "${DO_SPACES_SECRET}" ]]; then
  echo "Error: access key and secret key are required." >&2
  exit 1
fi

PREFIX="${PREFIX#/}"
S3_URI="s3://${BUCKET}"
if [[ -n "${PREFIX}" ]]; then
  S3_URI="${S3_URI}/${PREFIX}"
fi

mkdir -p "${DEST_DIR}"

echo "Syncing ${S3_URI} -> ${DEST_DIR}"
AWS_ACCESS_KEY_ID="${DO_SPACES_KEY}" \
AWS_SECRET_ACCESS_KEY="${DO_SPACES_SECRET}" \
AWS_DEFAULT_REGION="${REGION}" \
aws s3 sync "${S3_URI}" "${DEST_DIR}" \
  --endpoint-url "${ENDPOINT}" \
  --region "${REGION}" \
  --only-show-errors

unset DO_SPACES_KEY DO_SPACES_SECRET

echo "Done."
