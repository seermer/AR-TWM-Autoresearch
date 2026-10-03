"""Schemas the kernel validates every agent call against."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Ctx(BaseModel):
    model_config = ConfigDict(extra="allow")     # the kernel may add fields; agents ignore unknowns
    lineage: list[dict[str, Any]] = Field(default_factory=list)        # root .. parent
    siblings: list[dict[str, Any]] = Field(default_factory=list)       # the parent's finished children
    archive: dict[str, Any] = Field(default_factory=dict)
    nodes_remaining: int
    attempt: int
    max_attempts: int
    retry: dict[str, Any] | None = None          # the failed attempt's report, when retrying
    folders: dict[str, str] = Field(default_factory=dict)              # container path -> what it is for
    dry_run: bool = False


class EditContext(_Ctx):
    agent_dir: str = "/agent"


class RecipeContext(_Ctx):
    workspace: str = "/workspace"
    clip_pool_size: int = 0                                         # all clips in the archive
    clip_pool_stats: dict[str, Any] = Field(default_factory=dict)   # clips per ingesting node, source, format
    parent_data_commit: str | None = None
    parent_recipe: dict[str, Any] = Field(default_factory=dict)
    base_recipe: dict[str, Any] = Field(default_factory=dict)
    recipe_guide: dict[str, Any] = Field(default_factory=dict)   # tunable key -> base, meaning
    metric_guide: dict[str, Any] = Field(default_factory=dict)   # metric -> dimension, weight, what it measures
    tunable_rules: dict[str, Any] = Field(default_factory=dict)
    resolution_allowlist: list[list[int]] = Field(default_factory=list)
    lora_allowlist: list[list[int]] = Field(default_factory=list)
    n_gpus: int = 4
    tools: list[str] = Field(default_factory=list)   # enabled kernel tool names


class EditResult(BaseModel):
    summary: str = Field(min_length=1)


class RecipeResult(BaseModel):
    data_commit: str = Field(min_length=1)
    recipe: dict[str, Any]
    rationale: str = Field(min_length=1)


CONTEXT_MODELS = {"edit_self": EditContext, "improve_recipe": RecipeContext}
RESULT_MODELS = {"edit_self": EditResult, "improve_recipe": RecipeResult}
