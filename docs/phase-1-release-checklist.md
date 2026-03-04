# Phase 1 Release Checklist (Technical Users)

Use this checklist before publishing a new Phase 1 release.

## Code and quality

- [ ] `master` is up to date and clean.
- [ ] Critical flows validated locally:
  - [ ] dashboard launch
  - [ ] audio capture
  - [ ] shabad detection
  - [ ] STTM control
- [ ] Installer validated on a clean machine:
  - [ ] fresh install
  - [ ] reinstall/update path
  - [ ] uninstall path
- [ ] `scripts/install.sh` still matches current dependency/runtime needs.

## Versioning and release prep

- [ ] Bump version in `pyproject.toml` if needed.
- [ ] Prepare release notes from `docs/phase-1-release-notes-template.md`.
- [ ] Confirm README install command points to `master` and valid script path.
- [ ] Confirm known issues/workarounds are documented.

## GitHub release

- [ ] Create a Git tag: `vX.Y.Z`.
- [ ] Push tag to origin.
- [ ] Create GitHub Release with:
  - [ ] title `vX.Y.Z`
  - [ ] release notes
  - [ ] install command block
- [ ] Mark as pre-release while in Phase 1 beta.

## Distribution page and comms

- [ ] Landing page updated with latest version/changelog summary.
- [ ] Vercel deployment is green and serving latest commit.
- [ ] Share rollout message with:
  - [ ] install command
  - [ ] minimum prerequisites
  - [ ] support/contact path

## Rollback readiness

- [ ] Previous known-good tag is documented.
- [ ] Rollback steps tested:
  - [ ] reinstall pinned tag/commit
  - [ ] verify dashboard + detection after rollback
