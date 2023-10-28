from fastapi import APIRouter
from uuid import uuid4
from fastapi import status
from fastapi import Query
from src.models.schemas.image import ImageSchema
from src.pipelines.spatial_filtering.core import enhance_skeletons
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from src.utils.fs_io import img_to_base64
from typing import Sequence

from src.utils.images import get_image_schema

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
    img_in = open_img(dir_in, name)
    img_in_schemes = [get_image_schema(img_in, name)]
    pipeline_imgs = enhance_skeletons(img_in)
    pipeline_imgs_schemes = [
        get_image_schema(img, idx) for idx, img in enumerate(pipeline_imgs)
    ]
    return img_in_schemes + pipeline_imgs_schemes
