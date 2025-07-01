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
from graph_generation.logger.base import ProgressLogger
from graph_generation.storage.pipeline_storage import PipelineStorage
from graph_generation.utils.storage import write_table_to_storage

log = logging.getLogger(__name__)


async def run_workflow(
    config: graph_generationConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load and parse input documents into a standard format."""
    output = await load_input_documents(
        config.input,
        context.input_storage,
        context.progress_logger,
    )

    log.info("Final # of rows loaded: %s", len(output))
    context.stats.num_documents = len(output)

    await write_table_to_storage(output, "documents", context.output_storage)

    return WorkflowFunctionOutput(result=output)


async def load_input_documents(
    config: InputConfig, storage: PipelineStorage, progress_logger: ProgressLogger
) -> pd.DataFrame:
    """Load and parse input documents into a standard format."""
    return await create_input(config, storage, progress_logger)
