#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade build 'Nuitka>=2.7'
pyside6-deploy -c pysidedeploy.spec --force

APP="dist/LEAPS.app"
if [[ ! -d "$APP" ]]; then
  echo "Expected $APP was not produced" >&2
  exit 1
fi

if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp \
    --entitlements packaging/entitlements.plist \
    --sign "$APPLE_SIGNING_IDENTITY" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
fi

mkdir -p artifacts
hdiutil create -volname LEAPS -srcfolder "$APP" -ov -format UDZO artifacts/LEAPS-Apple-Silicon.dmg

if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --timestamp --sign "$APPLE_SIGNING_IDENTITY" artifacts/LEAPS-Apple-Silicon.dmg
fi

if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
  xcrun notarytool submit artifacts/LEAPS-Apple-Silicon.dmg --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
  xcrun stapler staple artifacts/LEAPS-Apple-Silicon.dmg
  spctl --assess --type open --context context:primary-signature --verbose artifacts/LEAPS-Apple-Silicon.dmg
fi

