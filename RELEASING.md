# Releasing Orchid Ranker

## Before the release

1. Start from a reviewed release candidate. Development work uses a PEP 440
   development version; the release candidate must use the final version that
   will appear on its tag (for this release, `1.0.0`).
2. Update `pyproject.toml`, `src/orchid_ranker/__init__.py`, and `CHANGELOG.md`
   together; all three identifiers must match for a tagged release.
3. Run the complete local gate:

   ```bash
   ./scripts/run_full_tests.sh
   ```

4. Build and inspect the distribution:

   ```bash
   python -m build
   python -m twine check dist/*
   ```

5. Confirm the wheel contains only the intended package files and that the
   source archive contains the current documentation and examples. Install the
   built wheel with dependencies into a clean environment that does not inherit
   system packages, then run the package smoke test before publication.

## Publish

Commit the reviewed release, tag it, and push the tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Publish the checked distribution through the project’s configured package
registry workflow. Do not rebuild artifacts after review; publish the files
that passed the release checks.

## After publishing

Verify the package can be installed in a clean environment, then update the
changelog with a new `Unreleased` section for the next change.
