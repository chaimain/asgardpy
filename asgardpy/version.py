"""
Asgardpy version
"""
MAJOR = "0"
MINOR = "4"
# On main and in a nightly release the patch should be one ahead of the last
# released build.
PATCH = "0"
# This is mainly for nightly builds which have the suffix ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
SUFFIX = ""

VERSION_SHORT = f"{MAJOR}.{MINOR}"
VERSION = f"{MAJOR}.{MINOR}.{PATCH}{SUFFIX}"
