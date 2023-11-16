# GitHub Release Process

## Steps

1. Open a new pull request to prepare the release. This should be the last pull
  request to be merged before making the actual release.

  Run `towncrier` in to render the changelog:

    ```
    git fetch
    git switch -c prepare_<VERSION NUMBER> origin/main
    towncrier build --version=<VERSION NUMBER>
    ```
