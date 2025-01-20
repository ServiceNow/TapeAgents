from typing import Annotated, TypeAlias

from pydantic import Field

FunctionName: TypeAlias = Annotated[str, Field(description="The name of a function.")]
ParameterName: TypeAlias = Annotated[str, Field(description="The name of a function parameter.")]
