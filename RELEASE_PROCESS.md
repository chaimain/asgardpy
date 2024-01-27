# GitHub Release Process

## Steps

1. Open a new pull request to prepare the release. This should be the last pull
  request to be merged before making the actual release.

  ```bash
  git fetch
  git switch -c prepare_<VERSION NUMBER> origin/main
  ```

2. Run `towncrier` in to render the changelog:

  ```bash
  towncrier build --version=<VERSION NUMBER>
  ```

3. Commit the changes made:

  ```bash
  git commit -a
  ```

4. Add TAG of the version number:

  ```bash
  git tag v<VERSION NUMBER> -m v<VERSION NUMBER>
  git push --tags
  ```
