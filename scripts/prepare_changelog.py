from datetime import datetime
from pathlib import Path

from asgardpy.version import base_version


def main():
    changelog = Path("CHANGELOG.md")

    with changelog.open() as f:
        lines = f.readlines()

    insert_index: int = -1
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("## [Unreleased]"):
            insert_index = i + 1
        elif line.startswith(f"## [v{base_version}]"):
            print("CHANGELOG already up-to-date")
            return
        elif line.startswith("## [v"):
            break

    if insert_index < 0:
        raise RuntimeError("Couldn't find 'Unreleased' section")

    lines.insert(insert_index, "\n")
    lines.insert(
        insert_index + 1,
        f"## [v{base_version}](https://github.com/chaimain/asgardpy/releases/tag/v{base_version}) - "
        f"{datetime.now().strftime('%Y-%m-%d')}\n",
    )

    with changelog.open("w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
