from fastapi import APIRouter
from fastapi import status
from fastapi import Query

from src.models.schemas.base import BaseSchemaModel
from src.models.schemas.image import ImageSchema
from src.pipelines.frequency_filtering.core import fft_smoothen_sharpen
from src.pipelines.spatial_filtering.core import enhance_skeletons
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from typing import Sequence
from src.models.schemas.base import BaseSchemaModel
from pydantic.types import List, UUID4
from src.utils.images import get_image_schema

router = APIRouter(prefix="/frequency-filtering", tags=["frequency-filtering"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "skeleton.jpg"


class Test(BaseSchemaModel):
    source: ImageSchema
    spectrum: ImageSchema


@router.get(
    path="/smoothening-sharpening",
    name="frequency-filtering:smoothening-sharpening",
    response_model=Sequence[ImageSchema],
    status_code=status.HTTP_200_OK,
)
async def smoothen_sharpen(
    name: str = Query(description="Image name", default=default_image),
):
    img_in = open_img(dir_in, name)
    pipeline_imgs = fft_smoothen_sharpen(img_in)
    return Test(
        source=get_image_schema(img_in, name),
        spectrum=get_image_schema(pipeline_imgs, "spectrum"),
    )
