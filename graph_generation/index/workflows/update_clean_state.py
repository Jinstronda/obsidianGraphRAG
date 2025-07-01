# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

from graph_generation.config.models.graph_rag_config import graph_generationConfig
from graph_generation.index.typing.context import PipelineRunContext
from graph_generation.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(  # noqa: RUF029
    _config: graph_generationConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Clean the state after the update."""
    logger.info("Cleaning State")
    keys_to_delete = [
        key_name
        for key_name in context.state
        if key_name.startswith("incremental_update_")
    ]

    for key_name in keys_to_delete:
        del context.state[key_name]

    return WorkflowFunctionOutput(result=None)
