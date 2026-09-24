from ar_contract.models import EditResult, RecipeResult


def edit_self(ctx):
    return EditResult(summary="nothing to change")


def improve_recipe(ctx):
    return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={}, rationale="dry run")
