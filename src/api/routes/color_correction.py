from fastapi import APIRouter
import base64
from io import BytesIO
from uuid import uuid4
from fastapi import status
from fastapi import Query
from loguru import logger
import numpy as np
from PIL import Image
from src.pipelines.image_description.methods import describe_channels
from src.pipelines.color_correction.methods import spline_correction
from src.pipelines.color_correction.core import make_correction
from src.models.schemas.image import ImageSchema, ImageHist
from src.models.schemas.color_correction import CorrectionOut
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR

router = APIRouter(prefix="/color-correction", tags=["color-correction"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs/02"
default_image = "normalization.png"


@router.get(
    path="/spline",
    name="color-correction:spline",
    response_model=CorrectionOut,
    status_code=status.HTTP_200_OK,
)
async def color_correction_spline(
    name: str = Query(description="Image name", default=default_image),
    xp: str = Query(description="X coordinates of interpolation points"),
    fp: str = Query(description="Y coordinates of interpolation points"),
):
    xp = [float(x) for x in xp.split(",")]  # todo move to dependency
    fp = [float(x) for x in fp.split(",")]
    img_in_id = uuid4()
    logger.info(
        "Processing image {img_id} spline correction {xp}, {fp}",
        img_id=img_in_id,
        xp=xp,
        fp=fp,
    )
    img_in = open_img(dir_in, name)

    ri_in, gi_in, bi_in = describe_channels(np.asarray(img_in))
    buffer_in = BytesIO()
    Image.fromarray(img_in).save(buffer_in, format="PNG")
    img_in_schema = ImageSchema(
        id=img_in_id,
        img_src="data:image/png;base64,"
        + base64.b64encode(buffer_in.getvalue()).decode("utf-8"),
        hist=ImageHist(r=ri_in, g=gi_in, b=bi_in),
    )

    img_out = make_correction(img_in, spline_correction, xp, fp)

    ri_out, gi_out, bi_out = describe_channels(np.asarray(img_out))
    buffer_out = BytesIO()
    Image.fromarray(img_out).save(buffer_out, format="PNG")
    img_out_schema = ImageSchema(
        id=uuid4(),
        img_src="data:image/png;base64,"
        + base64.b64encode(buffer_out.getvalue()).decode("utf-8"),
        hist=ImageHist(r=ri_out, g=gi_out, b=bi_out),
    )
    logger.info("Processing image {img_id} done", img_id=img_in_id)
    return CorrectionOut(img_in=img_in_schema, img_out=img_out_schema)
