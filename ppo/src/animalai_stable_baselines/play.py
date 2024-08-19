# ruff: noqa: E402
import logging
import sys
from pathlib import Path

# Make sure this is above the import of AnimalAIEnvironment
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s",
    level=logging.INFO,
)
from animalai.environment import AnimalAIEnvironment

import animalai_stable_baselines.utils as utils


def main():
    env = AnimalAIEnvironment(
        file_name=str(utils.find_env_path(Path("./aai/"))),
        arenas_configurations=str("./aai/configs/sanityGreen.yml"),
        play=True,
    )

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Closing environment...")
        env.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
