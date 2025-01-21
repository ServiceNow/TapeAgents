"""
Functions for parallel processing of agent workers.
"""

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Callable, Generator, Iterable, Literal, TypeVar

from .config import is_debug_mode

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


def sequential_processor(
    stream: Iterable[InputType],
    worker_func: Callable[[InputType], OutputType],
    n_workers: Literal[1] = 1,
    initializer: None | Callable[..., None] = None,
    initargs: tuple[Any, ...] = (),
) -> Generator[OutputType | Exception, None, None]:
    if n_workers != 1:
        raise ValueError(
            f"Sequential processing requires n_workers == 1, "
            f"got {n_workers}. Use threaded or multiprocessing-based "
            "processors for more workers."
        )

    if initializer is not None:
        initializer(*initargs)

    for item in stream:
        try:
            yield worker_func(item)
        except Exception as e:
            yield e


def process_pool_processor(
    stream: Iterable[InputType],
    worker_func: Callable[[InputType], OutputType],
    n_workers: int,
    initializer: None | Callable[..., None] = None,
    initargs: tuple[Any, ...] = (),
    keep_order=False,
) -> Generator[OutputType | Exception, None, None]:
    """
    Processes a stream of items using a process pool with push semantics.

    If n_workers is set to 0, the items are processed using a simple loop
    without multiprocessing (for debugging purposes).

    This function submits all tasks from the input stream to a process pool
    for concurrent processing. Unlike a thread pool, a process pool uses
    multiple processes, which can provide true parallelism especially useful
    for CPU-bound tasks. The tasks are executed immediately and run to
    completion irrespective of whether their results are consumed.

    Args:
        stream (Generator[InputType, None, None]): A generator that yields items to process.
        n_workers (int): The number of worker processes to use.
        worker_func (Callable[[InputType], OutputType]): The function to process each item. This
            function should be picklable since it will be passed across process boundaries.
        initializer (None | Callable[..., None]): An optional initializer function
            to call for each worker process. Useful for setting up process-specific resources.
        initargs (tuple[Any, ...]): Arguments to pass to the initializer.

    Yields:
        OutputType | Exception: The processed result for each input item, or an exception raised during processing
            of the corresponding input item. Results are yielded as they become available.

    Note:
        Since this uses multiple processes, the `worker_func` and the items in the stream
        should not rely on shared state with the main process unless that state is safe to
        share across processes (like using multiprocessing-safe constructs).
    """
    if n_workers == 0:
        # Process items using a simple loop without threading
        yield from sequential_processor(stream, worker_func, 1, initializer, initargs)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            futures: Iterable[Future[OutputType]] = [executor.submit(worker_func, a) for a in stream]
            if not keep_order:
                futures = as_completed(futures)
            for future in futures:
                try:
                    yield future.result()
                except Exception as e:
                    yield e


def lazy_thread_pool_processor(
    stream: Iterable[InputType],
    worker_func: Callable[[InputType], OutputType],
    n_workers: int,
    initializer: None | Callable[..., None] = None,
    initargs: tuple[Any, ...] = (),
    futures_queue_timeout: float = 1.0,
    producer_thread_shutdown_timeout: float = 10.0,
) -> Generator[OutputType | Exception, None, None]:
    """
    Processes a stream of items in a thread pool with pull semantics.

    If n_workers is set to 0, the items are processed using a simple loop
    without threading (for debugging purposes).

    Tasks from the input stream are submitted to the thread pool only when
    the main generator is asked for a value. This ensures tasks are executed
    on-demand, as their results are consumed.

    Args:
        stream (Generator[InputType, None, None]): InputType generator that yields items to process.
        n_workers (int): The number of worker threads to use.
        worker_func (Callable[[InputType], OutputType]): The function to process each item.
        initializer (Optional[Callable[..., None]]): An optional initializer function
            to call for each worker thread.
        initargs (tuple[Any, ...]): Arguments for the initializer.
        futures_queue_timeout (float): The timeout for the futures queue (default: 1.).
        producer_thread_shutdown_timeout (float): The timeout for the producer thread (default: 10.).

    Yields:
        OutputType | Exception: The processed result for each input item, or an exception raised during processing
            of the corresponding input item.
    """
    if n_workers == 0:
        # Process items using a simple loop without threading
        yield from sequential_processor(stream, worker_func, 1, initializer, initargs)
    else:
        with ThreadPoolExecutor(
            max_workers=n_workers,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            # Bounded queue to store futures
            futures_queue: Queue[Future[OutputType]] = Queue(maxsize=n_workers)

            # Events for inter-thread communication
            producer_done_event = Event()
            consumer_stopped_event = Event()

            # Producer thread function
            def producer():
                for item in stream:
                    if consumer_stopped_event.is_set():
                        return
                    # Submit task and add to the queue
                    future = executor.submit(worker_func, item)
                    # Use put with a timeout to periodically check consumer_stopped_event
                    while True:
                        try:
                            futures_queue.put(future, timeout=futures_queue_timeout)
                            break
                        except Full:
                            if consumer_stopped_event.is_set():
                                return
                producer_done_event.set()

            producer_thread = Thread(target=producer)
            producer_thread.start()

            try:
                # Main (consumer) thread
                producer_was_done_before_get = None
                while True:
                    try:
                        # Get the next completed future
                        producer_was_done_before_get = producer_done_event.is_set()
                        future = futures_queue.get_nowait()
                        try:
                            yield future.result()
                        except Exception as e:
                            yield e
                    except Empty:
                        # If queue is empty and producer was done before we checked the queue,
                        # break out of loop
                        if producer_was_done_before_get:
                            break

            finally:
                # If consumer stops early, signal the producer to stop
                consumer_stopped_event.set()
                producer_thread.join(timeout=producer_thread_shutdown_timeout)

                if producer_thread.is_alive():
                    raise RuntimeError("Producer thread is still alive after timeout")


def eager_thread_pool_processor(
    stream: Iterable[InputType],
    worker_func: Callable[[InputType], OutputType],
    n_workers: int,
    initializer: None | Callable[..., None] = None,
    initargs: tuple[Any, ...] = (),
    ordered: bool = False,
) -> Generator[OutputType | Exception, None, None]:
    """
    Processes a stream of items in a thread pool with eager processing.

    Unlike lazy processing, this processor submits all tasks to the thread pool
    upfront and returns results as they complete.

    Args:
        stream: Input generator that yields items to process
        worker_func: Function to process each item
        n_workers: Number of worker threads
        initializer: Optional initializer function for worker threads
        initargs: Arguments for the initializer function
        ordered: If True, yield results in the order of input stream

    Yields:
        Processed results or exceptions for each input item
    """
    # Special case for no workers (sequential processing)
    if n_workers == 0:
        yield from (worker_func(item) for item in stream)
        return

    with ThreadPoolExecutor(max_workers=n_workers, initializer=initializer, initargs=initargs) as executor:
        # Submit all tasks upfront
        if ordered:
            # Preserve order of inputs
            futures = []
            for item in stream:
                futures.append(executor.submit(worker_func, item))

            # Yield results in original order
            for future in futures:
                try:
                    yield future.result()
                except Exception as e:
                    yield e
        else:
            # Yield results as they complete (out of order)
            futures = [executor.submit(worker_func, item) for item in stream]

            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    yield e


def choose_processor(n_workers: int):
    return (
        partial(eager_thread_pool_processor, n_workers=n_workers)
        if n_workers > 0 and not is_debug_mode()
        else partial(sequential_processor, n_workers=1)
    )
