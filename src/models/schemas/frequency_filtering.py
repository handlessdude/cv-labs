from src.models.schemas.base import BaseSchemaModel
from src.models.schemas.image import ImageSchema


class FilterApplicationSchema(BaseSchemaModel):
    filt: ImageSchema
    spectrum: ImageSchema
    img_out: ImageSchema


class FilteringPipelineSchema(BaseSchemaModel):
    smoothing_schema: FilterApplicationSchema
    sharpening_schema: FilterApplicationSchema
