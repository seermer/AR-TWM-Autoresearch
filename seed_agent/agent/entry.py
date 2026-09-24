"""Fixed entry points (spec 1.1.4): the kernel imports exactly edit_self and improve_recipe.
Also this agent's settings; the kernel sets the environment variables."""
import os

from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult

MODEL = os.environ.get("AR_DEFAULT_MODEL", "mock-model")
CONTEXT_WINDOW = int(os.environ.get("AR_CONTEXT_WINDOW", "128000"))   # the model's window, in tokens
COMPACT_AT = float(os.environ.get("AR_COMPACT_AT", "0.85"))           # auto-compact threshold
AGENT_ROOT = os.environ.get("AR_AGENT_DIR", "/agent")
WORKSPACE = os.environ.get("AR_WORKSPACE", "/workspace")
CHECK_ROUNDS = 3             # build -> recipe -> recipe_check loops inside one attempt
SELFTEST_ROUNDS = 2          # implement -> self-test loops inside one attempt
BRIEF_CHARS = 60_000         # cap on the JSON context handed to a role


async def edit_self(ctx: EditContext) -> EditResult:
    from .orchestration import run_meta      # imported here: orchestration reads the settings above
    return await run_meta(ctx)


async def improve_recipe(ctx: RecipeContext) -> RecipeResult:
    from .orchestration import run_task
    return await run_task(ctx)
