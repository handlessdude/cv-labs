from fastapi import APIRouter
from uuid import uuid4
from fastapi import status
from fastapi import Query
from loguru import logger
import numpy as np
from src.pipelines.image_description.methods import describe_channels
from src.pipelines.quantization.methods import convert_to_halftone
from src.models.schemas.image import ImageSchema, ImageHist
from src.models.schemas.color_correction import CorrectionOut
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from src.utils.fs_io import img_to_base64

router = APIRouter(prefix="/quantization", tags=["quantization"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs/03"
default_image = "lisa.png"


@router.get(
    path="/halftone",
    name="quantization:halftone",
    response_model=CorrectionOut,
    status_code=status.HTTP_200_OK,
)
async def halftone(
    name: str = Query(description="Image name", default=default_image),
):
    img_in_id = uuid4()
    logger.info("Processing halftone for image {img_id}", img_id=img_in_id)

    img_in = open_img(dir_in, name)
    ri_in, gi_in, bi_in = describe_channels(np.asarray(img_in))
    img_in_schema = ImageSchema(
        id=img_in_id,
        img_src=img_to_base64(img_in),
        hist=ImageHist(r=ri_in, g=gi_in, b=bi_in),
    )

    img_out = convert_to_halftone(img_in)
    ri_out, gi_out, bi_out = describe_channels(np.asarray(img_out))
    img_out_schema = ImageSchema(
        id=uuid4(),
        img_src=img_to_base64(img_out),
        hist=ImageHist(r=ri_out, g=gi_out, b=bi_out),
    )
    logger.info("Processing image {img_id} done", img_id=img_in_id)
    return CorrectionOut(img_in=img_in_schema, img_out=img_out_schema)
