import time
import logging
import datetime
import torch
import os
import wandb
from pathlib import Path
from omegaconf import DictConfig
import json
from typing import Optional


logger = logging.getLogger(__name__)

class DistributedManager:
    @staticmethod
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    @staticmethod
    def get_world_size() -> int:
        return int(os.environ.get("WORLD_SIZE", "1"))
    
    @staticmethod
    def get_rank() -> int:
        return int(os.environ.get("RANK", "0"))

    @staticmethod
    def get_master_address() -> str:
        return os.environ.get("MASTER_ADDR", "localhost")
    
    @staticmethod
    def get_master_port() -> str:
        return os.environ.get("MASTER_PORT", "29500")
    
    @classmethod
    def cleanup_gpu_resources(cls):
        if torch.cuda.is_available():
            try:
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Monitor cleanup effectiveness
                cls.check_memory_status()
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

    @classmethod
    def robust_barrier(cls, message: str = "", timeout_mins: int = 30, max_retries: int = 3) -> bool:
        """Barrier implementation with retries"""
        logger.info(f"Activating barrier on rank {cls.get_rank()}")

        retry_delay = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"[Rank {cls.get_rank()}] Barrier attempt {attempt + 1}/{max_retries}: {message}")

                if cls.is_main_process():
                    torch.distributed.monitored_barrier(timeout=datetime.timedelta(minutes=timeout_mins), wait_all_ranks=True)
                else:
                    torch.distributed.monitored_barrier(timeout=datetime.timedelta(minutes=timeout_mins))

                logger.info(f"[Rank {cls.get_rank()}] Barrier successful: {message}")
                return True

            except Exception as e:
                logger.warning(f"[Rank {cls.get_rank()}] Barrier attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)

        logger.error(f"[Rank {cls.get_rank()}] Failed all barrier attempts: {message}")
        return False

    @classmethod
    def sync_nodes_file_based(cls, message: str, sync_dir: Path, timeout_mins: int = 30, rank: int = None, world_size: int = None) -> bool:
        """
        File-based sync for non-distributed phases.
        The aim is to maintain consistency between data generation and training phases.
        Raises RuntimeError if sync fails.
        """
        def retry_operation(operation, max_attempts=3, delay=1):
            for attempt in range(max_attempts):
                try:
                    return operation()
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    logger.warning(f"[Rank {rank}] Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            return False

        safe_message = message.replace(' ', '_')
        ready_file = sync_dir / f"rank_{rank}_ready_{safe_message}.json"
        
        try:
            retry_operation(lambda: os.makedirs(sync_dir, exist_ok=True))
            retry_operation(
                lambda: ready_file.touch(),
                max_attempts=10
            )
        except Exception as e:
            error_msg = f"[Rank {rank}] Failed to create sync file for {message}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"[Rank {rank}] Signaled ready for: {message}")
        
        # Wait for all ranks
        timeout = time.time() + (timeout_mins * 60)
        
        while time.time() < timeout:
            try:
                message_pattern = f"rank_*_ready_{safe_message}.json"
                valid_files = list(sync_dir.glob(message_pattern))
                
                if len(valid_files) == world_size:
                    logger.info(f"[Rank {rank}] All ranks ready for: {message}")
                    
                    # Only main process handles cleanup
                    if cls.is_main_process():
                        cleanup_success = retry_operation(
                            lambda: all(not f.exists() for f in [f.unlink() or f for f in valid_files]),
                            max_attempts=3
                        )
                        if not cleanup_success:
                            error_msg = f"[Rank {rank}] Failed to cleanup sync files for {message}"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                    
                    # All nodes wait a moment for cleanup to complete
                    time.sleep(2)
                    return True
                
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"[Rank {rank}] Error during sync for {message}: {e}"
                logger.error(error_msg)
                # Attempt to cleanup own file before raising
                try:
                    if ready_file.exists():
                        ready_file.unlink()
                except Exception as cleanup_e:
                    logger.error(f"[Rank {rank}] Additionally failed to cleanup own sync file: {cleanup_e}")
                raise RuntimeError(error_msg)
        
        # Timeout is a fatal error
        try:
            if ready_file.exists():
                ready_file.unlink()
        except Exception as e:
            logger.error(f"[Rank {rank}] Failed to cleanup own sync file: {e}")
        
        error_msg = f"[Rank {rank}] Timeout waiting for all ranks after {timeout_mins} minutes during {message}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @classmethod
    def sync_nodes(cls, message: str = "", timeout_mins: int = 30, sync_dir: Optional[Path] = None, rank: int = None, world_size: int = None) -> bool:
        """High-level sync function that chooses appropriate sync method"""
        if torch.distributed.is_initialized():
            logger.info(f"[Rank {rank}] Using distributed barrier for sync: {message}")
            return cls.robust_barrier(message, timeout_mins)
        elif sync_dir is not None:
            try:
                logger.info(f"[Rank {rank}] Using file-based sync with base path: {sync_dir}")
                return cls.sync_nodes_file_based(message, sync_dir, timeout_mins, rank, world_size)
                
            except Exception as e:
                logger.error(f"[Rank {rank}] Failed to setup file-based sync: {e}")
                raise RuntimeError(f"Sync failed on rank {rank}: {e}")
        else:
            error_msg = (
                f"[Rank {rank}] No synchronization method available! "
                "Either torch.distributed must be initialized or sync_dir must be provided"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @classmethod
    def check_memory_status(cls):
        """Check GPU memory status for debugging purposes."""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # Convert to GB
                max_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                
                logger.info(f"[Rank {cls.get_rank()}] GPU Memory Status:")
                logger.info(f"  - Allocated: {memory_allocated:.2f} GB")
                logger.info(f"  - Reserved:  {memory_reserved:.2f} GB")
                logger.info(f"  - Total:     {max_memory:.2f} GB")
                
                return {
                    "allocated": memory_allocated,
                    "reserved": memory_reserved,
                    "total": max_memory
                }
        except Exception as e:
            logger.error(f"[Rank {cls.get_rank()}] Failed to check memory status: {e}")
            return None


def init_wandb(
    cfg: DictConfig,
    run_dir: Path,
    config_for_wandb: DictConfig | dict,
    dist_manager: DistributedManager,
) -> wandb.sdk.wandb_run.Run:
    """Initialize W&B on the main process only."""
    if not dist_manager.is_main_process():
        logger.info(f"Skipping W&B init on non-main process")
        os.environ["WANDB_MODE"] = "offline"
        return

    try:
        from wandb.sdk import wandb_run

        # Set the port explicitly to avoid conflicts
        os.environ["WANDB_PORT"] = str(8080 + int(dist_manager.get_rank()))

        wandb.require("core")

        if config_for_wandb is None:
            config_for_wandb = cfg.dict()

        wandb_id = cfg.finetune.wandb_id

        if cfg.finetune.wandb_resume == "always":
            resume = True
        elif cfg.finetune.wandb_resume == "if_not_interactive":
            resume = not cfg.finetune.force_restart
        else:
            raise ValueError(f"Unknown value for wandb_resume: {cfg.finetune.wandb_resume}")

        wandb_name = run_dir.name if cfg.finetune.wandb_use_basename else str(run_dir)
        if len(wandb_name) > 128:
            logger.warning(f"wandb_name: {wandb_name} is longer than 128 characters. Truncating.")

        logger.info(f"Starting W&B init with name: {wandb_name[:128]}, resume: {resume}")

        run = wandb.init(
            name=wandb_name[:128],
            entity=cfg.finetune.wandb_entity_name,
            project=cfg.finetune.wandb_project_name,
            config=config_for_wandb,
            resume=resume,
            id=wandb_id,
            tags=cfg.finetune.tags,
        )

        logger.info("W&B initialization successful")

        if not isinstance(run, wandb_run.Run):
            raise RuntimeError("W&B init failed - returned object is not a Run")

        return run

    except Exception as e:
        raise RuntimeError(f"W&B initialization failed. Cannot continue without experiment tracking: {e}")


def flatten_dict_config(d: DictConfig | dict, separator=".") -> dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, DictConfig) or isinstance(v, dict):
            for sub_k, sub_v in flatten_dict_config(v).items():
                result[str(k) + separator + str(sub_k)] = sub_v
        else:
            result[k] = v
    return result