import datetime
from typing import Any
from uuid import UUID

import pydantic

from src.utils.formatting import snake_to_camel_case
from src.utils.datetime import datetime_into_isoformat


class BaseSchemaModel(pydantic.BaseModel):
    pass
    # def dict(self, **kwargs):
    #     result = super().dict(**kwargs)
    #
    #     def _convert_uuid_to_str(data: dict):
    #         for key, value in data.items():
    #             if isinstance(value, UUID):
    #                 data[key] = str(value)
    #         return data
    #
    #     return _convert_uuid_to_str(result)
    #
    # class Config(pydantic.BaseConfig):
    #     orm_mode: bool = True
    #     validate_assignment: bool = True
    #     use_enum_values = True
    #     allow_population_by_field_name: bool = True
    #     json_encoders: dict = {datetime.datetime: datetime_into_isoformat}
    #     alias_generator: Any = snake_to_camel_case
