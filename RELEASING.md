# Releasing Orchid Ranker

## Before the release

1. Choose the next semantic version.
2. Update `pyproject.toml`, `src/orchid_ranker/__init__.py`, and `CHANGELOG.md`.
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
   source archive contains the current documentation and examples.

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
