"""Schemas the kernel validates every agent call against."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Ctx(BaseModel):
    model_config = ConfigDict(extra="allow")     # the kernel may add fields; agents ignore unknowns
    lineage: list[dict[str, Any]] = Field(default_factory=list)        # root .. parent
    siblings: list[dict[str, Any]] = Field(default_factory=list)       # the parent's finished children
    archive: dict[str, Any] = Field(default_factory=dict)
    folders: dict[str, str] = Field(default_factory=dict)              # container path -> what it is for
    dry_run: bool = False


class EditContext(_Ctx):
    agent_dir: str = "/agent"


class RecipeContext(_Ctx):
    workspace: str = "/workspace"
    node_id: str = ""                                               # the node being built
    clip_pool_size: int = 0                                         # all clips in the archive
    clip_pool_stats: dict[str, Any] = Field(default_factory=dict)   # clips per ingesting node and source
    parent_data_commit: str | None = None
    parent_recipe: dict[str, Any] = Field(default_factory=dict)
    base_recipe: dict[str, Any] = Field(default_factory=dict)
    recipe_guide: dict[str, Any] = Field(default_factory=dict)   # tunable key -> base, meaning
    metric_guide: dict[str, Any] = Field(default_factory=dict)   # metric -> dimension, weight, what it measures
    target_skill: dict[str, str] = Field(default_factory=dict)   # the skill the run aims at: name, sentence, description
    tunable_rules: dict[str, Any] = Field(default_factory=dict)
    lora_allowlist: list[list[int]] = Field(default_factory=list)
    n_gpus: int = 4
    tools: list[str] = Field(default_factory=list)   # enabled kernel tool names


SUMMARY_CHARS = 4000


class _Result(BaseModel):
    snapshot_id: str = Field(min_length=1, description="the id of the snapshot that holds the work this phase "
                                                       "hands back")
    summary: str = Field(min_length=1, max_length=SUMMARY_CHARS,
                         description="what this node tested and built, the change it was expected to cause, and "
                                     "why this snapshot; stored with the node and shown to later nodes")


class EditResult(_Result):
    pass


class RecipeResult(_Result):
    pass


CONTEXT_MODELS = {"edit_self": EditContext, "improve_recipe": RecipeContext}
RESULT_MODELS = {"edit_self": EditResult, "improve_recipe": RecipeResult}
