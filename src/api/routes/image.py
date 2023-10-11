from fastapi import APIRouter
from uuid import uuid4
from fastapi import status
from fastapi import Query
import numpy as np
from src.pipelines.image_description.methods import describe_channels
from src.models.schemas.image import ImageSchema, ImageHist
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from src.utils.fs_io import img_to_base64

router = APIRouter(prefix="/image", tags=["image"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "gosling1.png"


@router.get(
    path="",
    name="image:get-image",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def halftone(
    name: str = Query(description="Image name", default=default_image),
):
    img_in_id = uuid4()
    img_in = open_img(dir_in, name)
    ri_in, gi_in, bi_in = describe_channels(np.asarray(img_in))
    return ImageSchema(
        id=img_in_id,
        name=name,
        img_src=img_to_base64(img_in),
        hist=ImageHist(r=ri_in, g=gi_in, b=bi_in),
    )
