import module_that_does_not_exist  # noqa: F401


def edit_self(ctx):
    return {"summary": "x"}


def improve_recipe(ctx):
    return {"data_commit": "c", "recipe": {}, "rationale": "r"}
