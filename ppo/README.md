# Recurrent PPO training

To execute the training, place the executable build for Animal-AI version 3.1.3 in the recurrent_ppo/env/aai/ folder

This project was configured using [Rye](https://rye-up.com/), but it should work with any tool supporting pyproject.toml.

**Note: Check the version requirements, they are quite particular.**  
**Note: Confirmed working with python == 3.9**  
**Note: Very modern GPUs might not be compatible because of the older Torch version.**

## Usage

- Playing `rye run play`
- Training `rye run train`

Check the [project.scripts] section of pyproject.toml for the relevant configuration.

## Resources

- [Docs](https://stable-baselines3.readthedocs.io/en/v2.0.0/) for Stable Baselines 3 ==v2.0.0.
