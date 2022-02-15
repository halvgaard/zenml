#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import Any, Dict, Tuple, Type, cast

from pydantic import BaseModel, Field
from pydantic.main import ModelMetaclass
from typing_extensions import Literal


class BaseTypedModelMeta(ModelMetaclass):
    """Metaclass responsible for adding type information to pydantic Models."""

    def __new__(
        mcs, name: str, bases: Tuple[Type[Any], ...], dct: Dict[str, Any]
    ) -> "BaseTypedModelMeta":
        """Creates a pydantic BaseModel class that includes a hidden attribute that
        reflects the full class identifier."""
        type_ann = Literal[f"{dct['__module__']}.{dct['__qualname__']}"]  # type: ignore
        type = Field(type_ann.__args__[0])
        dct.setdefault("__annotations__", dict())["type"] = type_ann
        dct["type"] = type
        cls = cast(
            Type["BaseTypedModel"], super().__new__(mcs, name, bases, dct)
        )
        return cls


class BaseTypedModel(BaseModel, metaclass=BaseTypedModelMeta):
    """Type pydantic Model base class.

    Use this class as a base class instead of BaseModel to automatically
    add a `type` literal attribute to the model that stores the name of the
    class.

    This can be useful when serializing models to JSON and then de-serializing
    them as part of a submodel union field, e.g.:

    ```python

    class BluePill(BaseTypedModel):
        ...

    class RedPill(BaseTypedModel):
        ...

    class TheMatrix(BaseTypedModel):
        choice: Union[BluePill, RedPill] = Field(..., discriminator='type')

    matrix = TheMatrix(choice=RedPill())
    d = matrix.dict()
    new_matrix = TheMatrix.parse_obj(d)
    assert isinstance(new_matrix.choice, RedPill)
    ```

    It can also facilitate de-serializing objects when their type isn't known:

    ```python
    type = d['type'].split('.')
    module = importlib.import_module('.'.join(type[:-1]))
    cls = getattr(module, type[-1])
    new_matrix = cls.parse_obj(d)
    assert isinstance(new_matrix.choice, RedPill)
    ```
    """
