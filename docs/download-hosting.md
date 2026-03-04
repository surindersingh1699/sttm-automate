# Download Hosting Strategy

This project should host downloadable desktop binaries through **GitHub Releases**.

## Why this is the best fit now

- Free and reliable hosting for release assets.
- Versioned URLs per tag (`vX.Y.Z`).
- Easy integration with landing page/API.
- Works well with existing GitHub Actions CI.

## What is implemented

- Tag-triggered workflow: `.github/workflows/release-binaries.yml`
- Builds:
  - macOS app bundle (zipped `.app`)
  - Windows `.exe`
- Publishes both files to the matching GitHub Release.

## Release flow

1. Merge changes to `master`.
2. Create and push a tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

3. Wait for workflow `Release Binaries` to complete.
4. Confirm release assets exist under GitHub Releases.
5. Landing page download buttons will auto-point to latest assets.

## Download URLs

Landing page reads from:

- `https://api.github.com/repos/surindersingh1699/sttm-automate/releases/latest`

Assets are selected by filename pattern:

- `STTM-Automate-mac-*.zip`
- `STTM-Automate-windows-*.exe`

## Optional upgrades later

- Code signing + notarization for macOS.
- Signed installer (`.dmg`/`.pkg`) instead of zipped `.app`.
- Windows Authenticode signing.
