import logging

from accelerate import Accelerator

logger = logging.getLogger(__name__)

# step_scheduler_with_optimizer=False prevents the scheduler
# from being stepped multiple times in the multi-gpu setting.
# (The default behavior in AcceleratedScheduler when split_batches=False is to
#   step() "num_processes" times, because they expect the lr schedule to
#   depend on processed samples/epochs, not completed_steps)
accelerator = Accelerator(step_scheduler_with_optimizer=False)
