from datetime import datetime
from pathlib import Path


def find_env_path(base: Path) -> Path:
    """
    Look for the latest version of the AAI.
    """
    error_msg = (
        "Could not automatically find any AAI environment binaries.\n\n"
        "We look in $BASE$/env/ for files matching {AAI,AnimalAI}.{x86_64,exe,app}, e.g. AAI.x86_64. "
        "Afterward, we look in the folders matching '$BASE$/env*/', "
        " taking the one that is lexicographically last, e.g. '$BASE$/env3.1.3/'.\n\n"
        "You can also specify the path exactly with the --env argument."
    ).replace("$BASE$", str(base))

    # Select folder
    env_folders = sorted(base.glob("env*"))
    if (base / "env/").exists():
        env_folders.append(base / "env/")
    if len(env_folders) == 0:
        reason = f"Could not find any folders matching {str(base)}/env*/"
        raise FileNotFoundError(f"{error_msg}\n\nReason: {reason}")
    env_folder = env_folders[-1]

    # Look for binary in selected folder
    # We brace expand manually because glob does not support it.
    binaries = [
        bin
        for bin_name in ["AAI", "AnimalAI"]
        for ext in ["x86_64", "exe", "app"]
        for bin in env_folder.glob(f"{bin_name}.{ext}")
    ]
    if len(binaries) == 0:
        reason = f"Could not find any AAI binaries in {env_folder}."
        raise FileNotFoundError(f"{error_msg}\n\nReason: {reason}")

    return binaries[0]

def make_logdir(logdir: Path, task: Path) -> Path:
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if logdir is not None:
        logdir = logdir
    else:
        logdir = Path("./aai/logdir/") / f'training-{date}'
    logdir.mkdir(parents=True, exist_ok=True)

    return logdir
