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

                # Attempt barrier with timeout
                torch.distributed.barrier(timeout=datetime.timedelta(minutes=timeout_mins))

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
    def sync_nodes_file_based(cls, message: str, base_path: Path, timeout_mins: int = 30) -> bool:
        """File-based synchronization for non-distributed phases"""
        rank = cls.get_rank()
        world_size = cls.get_world_size()
        sync_dir = base_path / "sync"
        
        # Ensure sync directory exists and is clean
        try:
            sync_dir.mkdir(exist_ok=True)
            # Clean any stale files from previous failed syncs for this message
            stale_files = list(sync_dir.glob(f"rank_*_ready_{message.replace(' ', '_')}.json"))
            for f in stale_files:
                try:
                    f.unlink()
                    logger.info(f"[Rank {rank}] Cleaned stale sync file: {f}")
                except Exception as e:
                    logger.warning(f"[Rank {rank}] Failed to clean stale file {f}: {e}")
        except Exception as e:
            logger.error(f"[Rank {rank}] Failed to setup sync directory: {e}")
            return False
        
        # Create our ready file with retry logic
        ready_file = sync_dir / f"rank_{rank}_ready_{message.replace(' ', '_')}.json"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(ready_file, 'w') as f:
                    json.dump({
                        "rank": rank,
                        "timestamp": time.time(),
                        "message": message,
                        "attempt": attempt + 1
                    }, f)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[Rank {rank}] Failed to create ready file after {max_retries} attempts: {e}")
                    return False
                logger.warning(f"[Rank {rank}] Attempt {attempt + 1} to create ready file failed: {e}")
                time.sleep(1)
        
        logger.info(f"[Rank {rank}] Signaled ready for: {message}")
        
        # Wait for all ranks with verification
        timeout = time.time() + (timeout_mins * 60)
        consecutive_complete_counts = 0
        required_consecutive_counts = 3  # Require multiple successful checks
        
        while time.time() < timeout:
            try:
                ready_files = list(sync_dir.glob(f"rank_*_ready_{message.replace(' ', '_')}.json"))
                
                # Verify file integrity
                valid_files = []
                for f in ready_files:
                    try:
                        with open(f, 'r') as file:
                            data = json.load(file)
                            if all(k in data for k in ["rank", "timestamp", "message"]):
                                valid_files.append(f)
                            else:
                                logger.warning(f"[Rank {rank}] Found malformed sync file: {f}")
                    except Exception as e:
                        logger.warning(f"[Rank {rank}] Failed to read sync file {f}: {e}")
                        continue
                
                if len(valid_files) == world_size:
                    consecutive_complete_counts += 1
                    if consecutive_complete_counts >= required_consecutive_counts:
                        logger.info(f"[Rank {rank}] All ranks ready for: {message} (verified {required_consecutive_counts} times)")
                        
                        # Clean up sync files with retry
                        cleanup_success = False
                        for cleanup_attempt in range(max_retries):
                            try:
                                for f in valid_files:
                                    f.unlink()
                                cleanup_success = True
                                logger.info(f"[Rank {rank}] Cleaned up sync files for: {message}")
                                break
                            except Exception as e:
                                logger.warning(f"[Rank {rank}] Cleanup attempt {cleanup_attempt + 1} failed: {e}")
                                time.sleep(1)
                        
                        if not cleanup_success:
                            logger.error(f"[Rank {rank}] Failed to clean up sync files after {max_retries} attempts")
                        
                        return True
                else:
                    consecutive_complete_counts = 0
                    
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"[Rank {rank}] Error during sync check: {e}")
                consecutive_complete_counts = 0
                time.sleep(1)
        
        logger.error(f"[Rank {rank}] Timeout waiting for all ranks to be ready for: {message}")
        return False

    @classmethod
    def sync_nodes(cls, message: str = "", timeout_mins: int = 30, base_path: Optional[Path] = None) -> bool:
        """High-level sync function that chooses appropriate sync method"""
        if torch.distributed.is_initialized():
            logger.warning(f"Distributed is initialized on rank {cls.get_rank()}")
            return cls.robust_barrier(message, timeout_mins)
        elif base_path is not None:
            logger.warning(f"Distributed is not initialized on rank {cls.get_rank()}, using file-based sync")
            return cls.sync_nodes_file_based(message, base_path, timeout_mins)
        else:
            logger.warning(f"No synchronization method available for message: {message}")
            return True

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