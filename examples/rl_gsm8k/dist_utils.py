import time
import logging
import datetime
import torch
import os

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
        """More robust barrier implementation with retries and cleanup"""
        if not torch.distributed.is_initialized():
            return True

        retry_delay = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"[Rank {cls.get_rank()}] Barrier attempt {attempt + 1}/{max_retries}: {message}")

                # Clear GPU memory before barrier
                cls.cleanup_gpu_resources()

                # Attempt barrier with timeout
                torch.distributed.barrier(timeout=datetime.timedelta(minutes=timeout_mins))

                logger.info(f"[Rank {cls.get_rank()}] Barrier successful: {message}")
                return True

            except Exception as e:
                logger.warning(f"[Rank {cls.get_rank()}] Barrier attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)

                    # Attempt to reinit process group
                    if cls.reinit_process_group(timeout_mins=timeout_mins):
                        logger.info("Successfully reinitialized process group")
                    else:
                        logger.warning("Failed to reinitialize process group")

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
    def all_gather_object(cls, obj):
        """Gather objects from all processes and return them in a list.
        
        Args:
            obj: Any picklable Python object
        Returns:
            list: List of objects gathered from all processes
        """
        if not torch.distributed.is_initialized():
            return [obj]
            
        try:
            gathered_objects = [None] * cls.get_world_size()
            torch.distributed.all_gather_object(gathered_objects, obj)
            return gathered_objects
            
        except Exception as e:
            logger.error(f"[Rank {cls.get_rank()}] Failed to gather objects: {e}")
            # Return list with just this process's object on failure
            return [obj]

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