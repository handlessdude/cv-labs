from typing import Any, List
from pydantic import BaseModel


class IOEntry(BaseModel):
    subfolder: str
    filename_in: str
    filename_out: str
    hist_in: str
    hist_out: str
    additional_args: List[Any]
