# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Pipeline workflow types."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from graph_generation.config.models.graph_rag_config import graph_generationConfig
from graph_generation.index.typing.context import PipelineRunContext


@dataclass
class WorkflowFunctionOutput:
    """Data container for Workflow function results."""

    result: Any | None
    """The result of the workflow function. This can be anything - we use it only for logging downstream, and expect each workflow function to write official outputs to the provided storage."""
    stop: bool = False
    """Flag to indicate if the workflow should stop after this function. This should only be used when continuation could cause an unstable failure."""


WorkflowFunction = Callable[
    [graph_generationConfig, PipelineRunContext],
    Awaitable[WorkflowFunctionOutput],
]
Workflow = tuple[str, WorkflowFunction]
