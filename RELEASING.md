# Releasing Orchid Ranker

## Before the release

1. Start from the next development version on `main` (currently
   `0.7.0.dev0`). Choose the final release version only for a reviewed release
   candidate.
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
   built wheel into a clean environment and run the package smoke test before
   publication.

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
