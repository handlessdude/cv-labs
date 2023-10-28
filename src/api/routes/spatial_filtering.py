from fastapi import APIRouter
from uuid import uuid4
from fastapi import status
from fastapi import Query
from src.models.schemas.image import ImageSchema
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from src.utils.fs_io import img_to_base64
from typing import Sequence

router = APIRouter(prefix="/spatial-filtering", tags=["spatial-filtering"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "skeleton.jpg"


@router.get(
    path="/skeleton-enhancement",
    name="spatial-filtering:skeleton-enhancement",
    response_model=Sequence[ImageSchema],
    status_code=status.HTTP_200_OK,
)
async def skeleton_enhancement(
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
    # pipeline_imgs = inspect_gears(img_in)
    # pipeline_imgs_schemes = [
    #     ImageSchema(id=uuid4(), name=name, img_src=img_to_base64(img))
    #     for img in pipeline_imgs
    # ]
    return img_in_schemes  # + pipeline_imgs_schemes
