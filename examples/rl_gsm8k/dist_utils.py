import time
import logging
import datetime
import torch
import os
import wandb
from pathlib import Path
from omegaconf import DictConfig

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
        """More robust barrier implementation with retries"""
        if not torch.distributed.is_initialized():
            return True

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
    def sync_nodes(cls, message: str = "", timeout_mins: int = 30) -> bool:
        """High-level sync function with additional safeguards"""
        if not torch.distributed.is_initialized():
            return True
            
        try:
            # Ensure GPU operations are finished
            cls.cleanup_gpu_resources()
            
            # Additional small wait to ensure all processes are ready
            time.sleep(cls.get_rank() * 0.1)  # Stagger by rank
            
            return cls.robust_barrier(message, timeout_mins)
            
        except Exception as e:
            logger.error(f"[Rank {cls.get_rank()}] Failed to sync nodes: {e}")
            return False

    @classmethod
    def broadcast_object(cls, obj, src=0):
        """Broadcast an object from the source rank to all other processes."""
        if not torch.distributed.is_initialized():
            return obj
            
        try:
            # Debug logging before broadcast
            logger.info(f"[Rank {cls.get_rank()}] Starting broadcast operation")
            
            # Ensure all processes are ready for broadcast
            if not cls.sync_nodes("before broadcast"):
                raise RuntimeError(f"Failed sync before broadcast on rank {cls.get_rank()}")
            
            # Create object list with None for non-source ranks
            object_list = [obj if cls.get_rank() == src else None]
            
            # Log object details before broadcast
            logger.info(f"[Rank {cls.get_rank()}] Before broadcast: "
                       f"object_list[0] type: {type(object_list[0])}, "
                       f"length: {len(object_list[0]) if object_list[0] and hasattr(object_list[0], '__len__') else 'N/A'}")
            
            # Perform broadcast with explicit timeout
            torch.distributed.broadcast_object_list(
                object_list,
                src=src,
                timeout=datetime.timedelta(minutes=10)
            )
            
            # Get result and verify
            result = object_list[0]
            
            # Log result details
            logger.info(f"[Rank {cls.get_rank()}] After broadcast: "
                       f"result type: {type(result)}, "
                       f"length: {len(result) if result and hasattr(result, '__len__') else 'N/A'}")
            
            # Verify broadcast result
            if result is None:
                raise RuntimeError(f"Broadcast resulted in None on rank {cls.get_rank()}")
            
            # Ensure all processes received the data
            if not cls.sync_nodes("after broadcast"):
                raise RuntimeError(f"Failed sync after broadcast on rank {cls.get_rank()}")
            
            logger.info(f"[Rank {cls.get_rank()}] Broadcast operation completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"[Rank {cls.get_rank()}] Failed to broadcast object: {e}")
            raise RuntimeError(f"Broadcast failed on rank {cls.get_rank()}: {e}")

    @classmethod
    def gather_object(cls, obj):
        """Gather objects from all processes to rank 0.

        Args:
            obj: Any picklable Python object
        Returns:
            list: On rank 0, list of objects gathered from all processes.
                  On other ranks, None.
        """
        if not torch.distributed.is_initialized():
            return [obj]

        try:
            if cls.get_rank() == '0':
                gathered_objects = [None] * cls.get_world_size()
            else:
                gathered_objects = None

            torch.distributed.gather_object(obj, gathered_objects, dst=0)
            return gathered_objects

        except Exception as e:
            logger.error(f"[Rank {cls.get_rank()}] Failed to gather objects: {e}")
            # Return list with just this process's object on rank 0
            return [obj] if cls.get_rank() == '0' else None

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