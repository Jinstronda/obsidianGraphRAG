# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utility functions for the graph_generation run module."""

from graph_generation.cache.memory_pipeline_cache import InMemoryCache
from graph_generation.cache.pipeline_cache import PipelineCache
from graph_generation.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graph_generation.callbacks.progress_workflow_callbacks import ProgressWorkflowCallbacks
from graph_generation.callbacks.workflow_callbacks import WorkflowCallbacks
from graph_generation.callbacks.workflow_callbacks_manager import WorkflowCallbacksManager
from graph_generation.config.models.graph_rag_config import graph_generationConfig
from graph_generation.index.typing.context import PipelineRunContext
from graph_generation.index.typing.state import PipelineState
from graph_generation.index.typing.stats import PipelineRunStats
from graph_generation.logger.base import ProgressLogger
from graph_generation.logger.null_progress import NullProgressLogger
from graph_generation.storage.memory_pipeline_storage import MemoryPipelineStorage
from graph_generation.storage.pipeline_storage import PipelineStorage
from graph_generation.utils.api import create_storage_from_config


def create_run_context(
    input_storage: PipelineStorage | None = None,
    output_storage: PipelineStorage | None = None,
    previous_storage: PipelineStorage | None = None,
    cache: PipelineCache | None = None,
    callbacks: WorkflowCallbacks | None = None,
    progress_logger: ProgressLogger | None = None,
    stats: PipelineRunStats | None = None,
    state: PipelineState | None = None,
) -> PipelineRunContext:
    """Create the run context for the pipeline."""
    return PipelineRunContext(
        input_storage=input_storage or MemoryPipelineStorage(),
        output_storage=output_storage or MemoryPipelineStorage(),
        previous_storage=previous_storage or MemoryPipelineStorage(),
        cache=cache or InMemoryCache(),
        callbacks=callbacks or NoopWorkflowCallbacks(),
        progress_logger=progress_logger or NullProgressLogger(),
        stats=stats or PipelineRunStats(),
        state=state or {},
    )


def create_callback_chain(
    callbacks: list[WorkflowCallbacks] | None, progress: ProgressLogger | None
) -> WorkflowCallbacks:
    """Create a callback manager that encompasses multiple callbacks."""
    manager = WorkflowCallbacksManager()
    for callback in callbacks or []:
        manager.register(callback)
    if progress is not None:
        manager.register(ProgressWorkflowCallbacks(progress))
    return manager


def get_update_storages(
    config: graph_generationConfig, timestamp: str
) -> tuple[PipelineStorage, PipelineStorage, PipelineStorage]:
    """Get storage objects for the update index run."""
    output_storage = create_storage_from_config(config.output)
    update_storage = create_storage_from_config(config.update_index_output)
    timestamped_storage = update_storage.child(timestamp)
    delta_storage = timestamped_storage.child("delta")
    previous_storage = timestamped_storage.child("previous")

    return output_storage, previous_storage, delta_storage
