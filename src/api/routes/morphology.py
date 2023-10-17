from fastapi import APIRouter
from uuid import uuid4
from fastapi import status
from fastapi import Query
from src.models.schemas.image import ImageSchema
from src.pipelines.morphology.core import inspect_gears
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from src.utils.fs_io import img_to_base64
from typing import Sequence

router = APIRouter(prefix="/morphology", tags=["morphology"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "gears.png"


@router.get(
    path="/gear-inspection",
    name="morphology:gear-inspection",
    response_model=Sequence[ImageSchema],
    status_code=status.HTTP_200_OK,
)
async def gear_inspection(
    name: str = Query(description="Image name", default=default_image),
):
    img_in_id = uuid4()
    img_in = open_img(dir_in, name)
    img_in_schemes = [
        ImageSchema(
            id=img_in_id,
            name=name,
            img_src=img_to_base64(img_in),
        )
    ]
    pipeline_imgs = inspect_gears(img_in)
    pipeline_imgs_schemes = [
        ImageSchema(id=uuid4(), name=name, img_src=img_to_base64(img))
        for img in pipeline_imgs
    ]
    return img_in_schemes + pipeline_imgs_schemes
