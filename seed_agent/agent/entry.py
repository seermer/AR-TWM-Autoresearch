"""Entry points and settings."""
import os

from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult

MODEL = os.environ.get("AR_DEFAULT_MODEL", "mock-model")
CONTEXT_WINDOW = int(os.environ.get("AR_CONTEXT_WINDOW", "1000000"))   # tokens
COMPACT_AT = float(os.environ.get("AR_COMPACT_AT", "0.6"))           # compaction threshold
AGENT_ROOT = os.environ.get("AR_AGENT_DIR", "/agent")
WORKSPACE = os.environ.get("AR_WORKSPACE", "/workspace")
MAX_ROUNDS = 4               # builder calls per phase
BRIEF_CHARS = 400_000        # chars of a role's context


async def edit_self(ctx: EditContext) -> EditResult:
    from .orchestration import run_meta
    return await run_meta(ctx)


async def improve_recipe(ctx: RecipeContext) -> RecipeResult:
    from .orchestration import run_task
    return await run_task(ctx)
