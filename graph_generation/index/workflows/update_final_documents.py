# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

from graph_generation.config.models.graph_rag_config import graph_generationConfig
from graph_generation.index.run.utils import get_update_storages
from graph_generation.index.typing.context import PipelineRunContext
from graph_generation.index.typing.workflow import WorkflowFunctionOutput
from graph_generation.index.update.incremental_index import concat_dataframes

logger = logging.getLogger(__name__)


async def run_workflow(
    config: graph_generationConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update the documents from a incremental index run."""
    logger.info("Updating Documents")
    output_storage, previous_storage, delta_storage = get_update_storages(
        config, context.state["update_timestamp"]
    )

    final_documents = await concat_dataframes(
        "documents", previous_storage, delta_storage, output_storage
    )

    context.state["incremental_update_final_documents"] = final_documents

    return WorkflowFunctionOutput(result=None)
