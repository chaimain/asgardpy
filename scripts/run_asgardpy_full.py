import argparse
import logging

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.config import AsgardpyConfig

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run Asgardpy")

parser.add_argument(
    "--config",
    "-c",
    help="Path to the Config file",
)


def main():
    args = parser.parse_args()
    base_config = AsgardpyConfig()

    main_config = base_config.read(args.config)
    log.info(f"Analysis steps mentioned in the config file: {main_config.general.steps}")
    log.info(f"Target source is: {main_config.target.source_name}")

    analysis = AsgardpyAnalysis(main_config)
    analysis.log = log

    analysis.run()


if __name__ == "__main__":
    main()
