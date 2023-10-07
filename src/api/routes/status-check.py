from fastapi import APIRouter
import base64
from io import BytesIO
from uuid import uuid4
from fastapi import status
from fastapi import Query
import numpy as np
from PIL import Image
from src.pipelines.image_description.methods import describe_channels
from src.models.schemas.image import ImageSchema, ImageHist
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR

router = APIRouter(prefix="/status-check", tags=["status-check"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs/02"
default_image = "normalization.png"


@router.get(path="/")
async def sanity_check():
    return {"status": "healthy"}


@router.get(
    path="/image",
    name="image:sample",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def get_image(
    name: str = Query(description="Image name", default=default_image),
):
    img_id = uuid4()
    img = open_img(dir_in, name)
    r_intensities, g_intensities, b_intensities = describe_channels(np.asarray(img))
    buffer = BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    return ImageSchema(
        id=img_id,
        img_src="data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("utf-8"),
        hist=ImageHist(r=r_intensities, g=g_intensities, b=b_intensities),
    )
