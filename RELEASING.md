# Release Process

## Versioning Policy

Orchid Ranker follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes and improvements

Current version is in `pyproject.toml` and `orchid_ranker/__init__.py`.

## Pre-Release Checklist

Before releasing, ensure:

1. All CI checks pass (GitHub Actions green)
2. `CHANGELOG.md` is updated with the new version and changes
3. Version is bumped in:
   - `pyproject.toml` (`version = "X.Y.Z"`)
   - `orchid_ranker/__init__.py` (`__version__ = "X.Y.Z"`)
4. Code is committed and pushed to main

## How to Release

Tag the release and push to GitHub:

```bash
git tag v0.3.0
git push origin v0.3.0
```

The tag name must follow the pattern `v*` (e.g., `v0.3.0`, `v1.0.0`).

## Automated Release Pipeline

The GitHub Actions workflow automatically:

1. **Build**: Creates source distribution (sdist) and wheel
2. **Validate**: Checks distributions with twine
3. **PyPI Publish**: Publishes to PyPI using Trusted Publisher (OIDC)
4. **GitHub Release**: Creates a GitHub Release with:
   - Wheel and source distribution
   - Software Bill of Materials (SBOM) in SPDX format
5. **Documentation**: Deploys versioned docs to GitHub Pages using `mike`

## Post-Release

After a release:

1. Bump to next development version: `X.Y.Z+1.dev0`
2. Commit and push to main
3. Update `CHANGELOG.md` with a new "Unreleased" section

## Troubleshooting

**PyPI publish fails**: Check that PyPI Trusted Publisher is configured for the repository. See [PyPI documentation](https://docs.pypi.org/trusted-publishers/).

**Docs deploy fails**: Ensure `gh-pages` branch exists. Create it with `git switch --orphan gh-pages && git commit --allow-empty -m "Initial commit"`.

**SBOM generation fails**: Update `anchore/sbom-action` to the latest version in `release.yml`.
