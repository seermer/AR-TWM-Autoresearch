"""Fixed entry points: the kernel imports exactly edit_self and improve_recipe.
Also this agent's settings; the kernel sets the environment variables."""
import os

from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult

MODEL = os.environ.get("AR_DEFAULT_MODEL", "mock-model")
CONTEXT_WINDOW = int(os.environ.get("AR_CONTEXT_WINDOW", "1000000"))   # the model's window, in tokens
COMPACT_AT = float(os.environ.get("AR_COMPACT_AT", "0.6"))           # auto-compact threshold
AGENT_ROOT = os.environ.get("AR_AGENT_DIR", "/agent")
WORKSPACE = os.environ.get("AR_WORKSPACE", "/workspace")
MAX_ROUNDS = 4               # plans per phase: the first and up to three more (asked for, or after a review)
BRIEF_CHARS = 400_000        # safety cap on the context digest handed to a role; its own caps keep it under ~350,000


async def edit_self(ctx: EditContext) -> EditResult:
    from .orchestration import run_meta      # imported here: orchestration reads the settings above
    return await run_meta(ctx)


async def improve_recipe(ctx: RecipeContext) -> RecipeResult:
    from .orchestration import run_task
    return await run_task(ctx)
