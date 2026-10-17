from ar_contract.models import EditResult, RecipeResult


def edit_self(ctx):
    return EditResult(snapshot_id="dry-run", summary="nothing to change")


def improve_recipe(ctx):
    return RecipeResult(snapshot_id="dry-run", summary="dry run")
