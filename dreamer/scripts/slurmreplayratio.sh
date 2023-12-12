sbatch <<EOT
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name="rr2"
#SBATCH --output=X%j-%x.out
singularity run \
  --nv \
  --env-file .env \
  dreamerv3-animalai.sif \
  ./scripts/replayratio.sh 2
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name="rr4"
#SBATCH --output=X%j-%x.out
singularity run \
  --nv \
  --env-file .env \
  dreamerv3-animalai.sif \
  ./scripts/replayratio.sh 4
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name="rr8"
#SBATCH --output=X%j-%x.out
singularity run \
  --nv \
  --env-file .env \
  dreamerv3-animalai.sif \
  ./scripts/replayratio.sh 8
EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --job-name="rr16"
# #SBATCH --output=X%j-%x.out
# singularity run \
#   --nv \
#   --env-file .env \
#   dreamerv3-animalai.sif \
#   ./scripts/replayratio.sh 16
# EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --job-name="rr32"
# #SBATCH --output=X%j-%x.out
# singularity run \
#   --nv \
#   --env-file .env \
#   dreamerv3-animalai.sif \
#   ./scripts/replayratio.sh 32
# EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --job-name="rr64"
# #SBATCH --output=X%j-%x.out
# singularity run \
#   --nv \
#   --env-file .env \
#   dreamerv3-animalai.sif \
#   ./scripts/replayratio.sh 64
# EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --job-name="rr128"
# #SBATCH --output=X%j-%x.out
# singularity run \
#   --nv \
#   --env-file .env \
#   dreamerv3-animalai.sif \
#   ./scripts/replayratio.sh 128
# EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --job-name="rr256"
# #SBATCH --output=X%j-%x.out
# singularity run \
#   --nv \
#   --env-file .env \
#   dreamerv3-animalai.sif \
#   ./scripts/replayratio.sh 256
# EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --job-name="rr512"
# #SBATCH --output=X%j-%x.out
# singularity run \
#   --nv \
#   --env-file .env \
#   dreamerv3-animalai.sif \
#   ./scripts/replayratio.sh 512
# EOT