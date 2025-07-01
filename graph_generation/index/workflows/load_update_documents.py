# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import pandas as pd

from graph_generation.config.models.graph_rag_config import graph_generationConfig
from graph_generation.config.models.input_config import InputConfig
from graph_generation.index.input.factory import create_input
from graph_generation.index.typing.context import PipelineRunContext
from graph_generation.index.typing.workflow import WorkflowFunctionOutput
from graph_generation.index.update.incremental_index import get_delta_docs
from graph_generation.logger.base import ProgressLogger
from graph_generation.storage.pipeline_storage import PipelineStorage
from graph_generation.utils.storage import write_table_to_storage

log = logging.getLogger(__name__)


async def run_workflow(
    config: graph_generationConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load and parse update-only input documents into a standard format."""
    output = await load_update_documents(
        config.input,
        context.input_storage,
        context.previous_storage,
        context.progress_logger,
    )

    log.info("Final # of update rows loaded: %s", len(output))
    context.stats.update_documents = len(output)

    if len(output) == 0:
        log.warning("No new update documents found.")
        context.progress_logger.warning("No new update documents found.")
        return WorkflowFunctionOutput(result=None, stop=True)

    await write_table_to_storage(output, "documents", context.output_storage)

    return WorkflowFunctionOutput(result=output)


async def load_update_documents(
    config: InputConfig,
    input_storage: PipelineStorage,
    previous_storage: PipelineStorage,
    progress_logger: ProgressLogger,
) -> pd.DataFrame:
    """Load and parse update-only input documents into a standard format."""
    input_documents = await create_input(config, input_storage, progress_logger)
    # previous storage is the output of the previous run
    # we'll use this to diff the input from the prior
    delta_documents = await get_delta_docs(input_documents, previous_storage)
    return delta_documents.new_inputs
