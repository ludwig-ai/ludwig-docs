# 1. Determine the version name

!!! note

    Version names always begin with `v`.

Examples of version names:

```python
"vX.Y"      # Release major version X (starts at 0), minor version Y (starts at 1).
"vX.YrcZ"   # Release candidate Z, without a period. (starts at 1)
"vX.Y.dev"  # Developer version, with a period.
```

Inspiration:

- [Python pre-releases](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#pre-release-versioning).
- [PEP0440 pre-releases](https://www.python.org/dev/peps/pep-0440/#pre-releases).

# 2. Update Ludwig versions in code

Create a new branch.

```bash
git checkout -b ludwig_release
git push --set-upstream origin ludwig_release
```

Update the versions referenced in globals and setup. [Reference PR](https://github.com/ludwig-ai/ludwig/pull/1723/files).

```
git commit -m "Update ludwig version to vX.YrcZ."
git push
```

Create a PR with the change (merge ludwig_release -> master).

Get approval from a [Ludwig maintainer](https://github.com/orgs/ludwig-ai/teams/ludwig-maintainers).

Merge PR (with squashing).

# 3. Tag the latest commit, and push the tag

After merging the PR from step 2, the latest commit on master should be the PR that upgrades ludwig versions in code.

In master:

```bash
git checkout master
git pull
```

Add a tag to the commit locally:

```bash
git tag -a vX.YrcZ -m "Ludwig vX.YrcZ"
git push --follow-tags
```

# 4. In Github, go to releases and "Draft a new release"

Loom [walk-through](https://www.loom.com/share/78eb7f9134404a80bde9359cfa7af2b7).

Release candidates don't need release notes. Full releases should have detailed release notes. Refer to past releases of
Ludwig ([example](https://github.com/ludwig-ai/ludwig/releases/tag/v0.4.1)) for a good structure to use.

Do not upload assets manually. These will be created automatically by Github.

For release candidates, check "pre-release".

# 5. Click publish

When the release notes are ready, click `Publish release` on Github. Ludwig's CI will automatically update PyPI.

# 6. Spread the word

Consider sharing the release on LinkedIn/Twitter.

# Appendix

## Oops, more PRs were merged after the version bump

If there were some last minute PRs merged after the version bump, reorder the commits to make the version bump be the last commit that gets tagged before the release.

[Reordering Commits in Git](https://www.youtube.com/watch?v=V9KpcGO7nLo)

## Oops, I tagged the wrong commit, and pushed it to github already

```bash
git tag -d <tagname>                  # delete the old tag locally
git push origin :refs/tags/<tagname>  # delete the old tag remotely
git tag <tagname> <commitId>          # make a new tag locally
git push origin <tagname>             # push the new local tag to the remote
```
