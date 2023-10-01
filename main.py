import base64
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel
from pydantic import UUID4
from uuid import uuid4
from fastapi import Depends, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import ValidationException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from loguru import logger
from utils.image_hist import describe_channels
from utils.dependencies import parse_list
from utils.io.fs import make_path, open_img
from numba import njit
import numpy as np
from PIL import Image
from utils.color_correction import make_correction, spline_correction

logger.add("app.log", rotation="500 MB", retention="7 days", level="INFO")


app = FastAPI(
  title="CV-labs"
)


origins = [
    "http://localhost:9000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=[
      "Content-Type",
      "Set-Cookie",
      "Access-Control-Allow-Headers",
      "Access-Control-Allow-Origin",
      "Authorization",
    ],
)


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


@app.get(path="/")
async def sanity_check():
    return {
      "status": 200
    }


class ImageHist(BaseModel):
    r: List[int]
    g: List[int]
    b: List[int]


class ImageSchema(BaseModel):
    id: UUID4
    img_src: str
    hist: ImageHist


class CorrectionOut(BaseModel):
    img_in: ImageSchema
    img_out: ImageSchema


dir_in = './inputs/02'
default_image = 'normalization.png'


@app.get(
    path="/color-correction/spline",
    name="color-correction:spline",
    response_model=CorrectionOut,
    status_code=status.HTTP_200_OK,
)
async def color_correction_spline(
    name: str = Query(description="Image name", default=default_image),
    xp: str = Query(description="X coordinates of interpolation points"),
    fp: str = Query(description="Y coordinates of interpolation points"),
):
    xp = [float(x) for x in xp.split(',')]  # todo move to dependency
    fp = [float(x) for x in fp.split(',')]
    img_in_id = uuid4()
    logger.info("Processing image {img_id} spline correction {xp}, {fp}", img_id=img_in_id, xp=xp, fp=fp)
    img_in = open_img(dir_in, name)
    ri_in, gi_in, bi_in = describe_channels(np.asarray(img_in))
    buffer_in = BytesIO()
    Image.fromarray(img_in).save(buffer_in, format="PNG")
    img_in_schema = ImageSchema(
        id=img_in_id,
        img_src="data:image/png;base64," + base64.b64encode(buffer_in.getvalue()).decode("utf-8"),
        hist=ImageHist(r=ri_in, g=gi_in, b=bi_in)
    )

    img_out = make_correction(img_in, spline_correction, xp, fp)
    ri_out, gi_out, bi_out = describe_channels(np.asarray(img_out))
    buffer_out = BytesIO()
    Image.fromarray(img_out).save(buffer_out, format="PNG")
    img_out_schema = ImageSchema(
        id=uuid4(),
        img_src="data:image/png;base64," + base64.b64encode(buffer_out.getvalue()).decode("utf-8"),
        hist=ImageHist(r=ri_out, g=gi_out, b=bi_out)
    )
    logger.info("Processing image {img_id} done", img_id=img_in_id)
    return CorrectionOut(
        img_in=img_in_schema,
        img_out=img_out_schema
    )


@njit(cache=True)
@app.get(
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
        img_src="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8"),
        hist=ImageHist(r=r_intensities, g=g_intensities, b=b_intensities)
    )

