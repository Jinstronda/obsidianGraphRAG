# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, Field

from graph_generation.config.defaults import graph_generation_config_defaults


class BasicSearchConfig(BaseModel):
    """The default configuration section for Cache."""

    prompt: str | None = Field(
        description="The basic search prompt to use.",
        default=graph_generation_config_defaults.basic_search.prompt,
    )
    chat_model_id: str = Field(
        description="The model ID to use for basic search.",
        default=graph_generation_config_defaults.basic_search.chat_model_id,
    )
    embedding_model_id: str = Field(
        description="The model ID to use for text embeddings.",
        default=graph_generation_config_defaults.basic_search.embedding_model_id,
    )
    k: int = Field(
        description="The number of text units to include in search context.",
        default=graph_generation_config_defaults.basic_search.k,
    )
    max_context_tokens: int = Field(
        description="The maximum tokens.",
        default=graph_generation_config_defaults.basic_search.max_context_tokens,
    )
