# Random and Heuristic Agents

This sub-directory runs the simulations of the reference agents (random, heuristic) for the experiments presented in the paper.

It uses `Rye` as a package manager, but any package manager that can read a `.toml` and lock files should work (e.g., Poetry, Pixi).

To run the simulations using Rye, do the following:
1. Change the directory to this folder (`/random_heuristic_agents/`).
2. Run `rye sync` to install the venv and packages.
3. Run `rye run random` to run the random agent simulations for the foraging task, the operant chamber task, and the what where when task.
4. Run `rye run heuristic` to run the heuristic agent simulations for the foraging task, the operant chamber task, and the what where when task.
5. Run `rye run competition_random_heuristic` to run the random and heuristic agent simulations for the competition. *Note*: the script will increment through ports to initialize AAI. If the port is occupied, the simulation will crash. Results up to that configuration file are stored in a csv. Simply restart the simulation having deleted the already run competition configuration files in `../configs/competition/`.

Shell scripts are also provided in `random_heuristic_agents/scripts/`.
