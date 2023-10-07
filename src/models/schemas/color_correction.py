from src.models.schemas.image import ImageSchema
from src.models.schemas.base import BaseSchemaModel


class CorrectionOut(BaseSchemaModel):
    img_in: ImageSchema
    img_out: ImageSchema
