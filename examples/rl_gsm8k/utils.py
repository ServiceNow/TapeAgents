"""
A light wrapper around the make job command.
"""
import logging
import subprocess
from dataclasses import dataclass, fields

from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration to pass to `make job`.

    If the default for a setting is None, it means that the default from Makefile will be used.
    Unless the respective environment variable is set to a non-empty value (this often happens
    when a script invoked with `make job` invokes another script with `make job`).

    """
    job_account: str | None = None
    gpu: int | None = None
    gpu_mem: int | None = None
    cpu: int | None = None
    cpu_mem: int | None = None
    nproc: int | None = None
    snapshot: bool = True
    dry_run: bool = False
    local: bool = False
    accelerate: bool = False
    deepspeed: bool = False
    conda: bool = True
    conda_exe: str | None = None
    env: str | None = None
    job_name: str | None = None  # Must be lowercase, without "-". Default name will be $(USER)$(RAND_ID) as described in a Makefile
    fp: str | None = None # default would be "bf16". Use "no" for half-precision training

    def to_env_variable_str(self):
        """Create a string that sets corresponding environment variables."""
        var_settings = []
        for field in fields(self):
            field_value = getattr(self, field.name)
            if field_value is not None:
                var_name = field.name.upper()
                if field.type == bool:
                    var_value = "1" if field_value else "0"
                else:
                    var_value = str(field_value)
                var_settings.append(f"{var_name}={var_value}")
        return " ".join(var_settings)


cs = ConfigStore.instance()
cs.store(group="job", name="basic", node=JobConfig)


def make_job(command: str, job_config: JobConfig):
    """
    Python interface to `make job` command.

    Arguments:
        script: path to script to run
        job_config: JobConfig object, specifies gpu, cpu, memory, etc.
        call_make: actually call `make` or just return the command (used to test only)

    Example of doing hyperparameter search:

    >>> for batch_size in [4, 8]:
    ...     for learning_rate in [1e-4, 1e-5]:
    ...         command = f'scripts/finetune.py finetune.train_batch_size={batch_size} finetune.learning_rate={learning_rate}'
    ...         job_config = JobConfig(gpu=1, gpu_mem=80, dry_run=True, local=False)
    ...         make_job(command, job_config=job_config, call_make=False)
    'make job GPU=1 GPU_MEM=80 SNAPSHOT=1 DRY_RUN=1 LOCAL=0 ACCELERATE=0 COMMAND="scripts/finetune.py finetune.train_batch_size=4 finetune.learning_rate=0.0001"'
    'make job GPU=1 GPU_MEM=80 SNAPSHOT=1 DRY_RUN=1 LOCAL=0 ACCELERATE=0 COMMAND="scripts/finetune.py finetune.train_batch_size=4 finetune.learning_rate=1e-05"'
    'make job GPU=1 GPU_MEM=80 SNAPSHOT=1 DRY_RUN=1 LOCAL=0 ACCELERATE=0 COMMAND="scripts/finetune.py finetune.train_batch_size=8 finetune.learning_rate=0.0001"'
    'make job GPU=1 GPU_MEM=80 SNAPSHOT=1 DRY_RUN=1 LOCAL=0 ACCELERATE=0 COMMAND="scripts/finetune.py finetune.train_batch_size=8 finetune.learning_rate=1e-05"'

    """
    if "'" in command:
        logger.warning("using single quote in your make_job command can be dangerous")
    make_command = f"make job {job_config.to_env_variable_str()} COMMAND='{command}'"
    return subprocess.call(make_command, shell=True)


def hydra_overrides(dict_: dict, escape_as_value=False):
    """Format a dictionary as hydra overrides

    When `escape_as_value` is set to True, the resulting
    string will be wrapped into
    - double quotes (") to protect it from Bash argument parsing
    - escaped double quotes (\\") to protect the value from Hydra parsing
      (escaping is necessary to pash the double-quotes to Hydra instead of Bash)

    """
    overrides = " ".join([f"{k}={v}" for k, v in dict_.items()])
    if escape_as_value:
        return f'"\\"{overrides}\\""'
    else:
        return overrides
