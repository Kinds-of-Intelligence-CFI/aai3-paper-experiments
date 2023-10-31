#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --gpus=0
#SBATCH --job-name="integration-test"
#SBATCH --output=X%j-%x.out
singularity --nv dreamerv3-animalai.sif python tests/integration.py
EOT