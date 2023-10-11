from src.models.schemas.base import BaseSchemaModel
from pydantic.types import List, UUID4


class ImageHist(BaseSchemaModel):
    r: List[int]
    g: List[int]
    b: List[int]


class ImageSchema(BaseSchemaModel):
    id: UUID4
    name: str
    img_src: str
    hist: ImageHist
