# Uploading this release

Target repository: https://github.com/MichaelW17/OPMS

The release folder is a local Git checkout based on the repository's original main commit. The existing Apache-2.0 LICENSE is preserved. Network access was verified, but Git could not find a usable write credential; the upload attempt stopped before changing the remote.

After configuring your GitHub authentication, run from this folder:

```bash
git push origin main
```

If another commit was added to the remote meanwhile, fetch and integrate it first. The release does not require a force push.

If starting from the ZIP, clone the existing repository, copy the extracted release contents into the clone, commit them, then push. GitHub Desktop can perform the same clone, copy, commit and push workflow. The ZIP excludes local Git metadata, temporary simulations and Python caches.

Raw powers are compressed into eight NPZ archives; the largest file is approximately 42.4 MB. Use Git or GitHub Desktop for the complete dataset rather than uploading individual files through the website. The folder contains approximately 487 MB of archived data plus source code and rendered results.
