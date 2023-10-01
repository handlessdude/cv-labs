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


class CorrectionOut(BaseModel):
    id: UUID4
    img: str
    hist: List[int]


@app.get(
    path="/color-correction/spline",
    name="color-correction:spline",
    response_model=CorrectionOut,
    status_code=status.HTTP_200_OK,
)
async def color_correction_spline(
  # img: str = Query(description="Image source in base64 format"),
  xp: List[float] = Depends(parse_list),
  fp: List[float] = Depends(parse_list),
):
    img_id = uuid4()
    logger.info("Processing image {img_id} with spline {xp}, {fp}", img_id=img_id, xp=xp, fp=fp)
    logger.info("Processing image {img_id} done", img_id=img_id)
    return {
      "id": img_id,
      "img": "",
      "hist": [1, 2, 3],
    }

dir_in = './inputs/02'
default_image = 'normalization.png'


class ImageHist(BaseModel):
    r: List[int]
    g: List[int]
    b: List[int]


class ImageOut(BaseModel):
    id: UUID4
    img: str
    hist: ImageHist

@njit(cache=True)
@app.get(
    path="/image",
    name="image:sample",
    response_model=ImageOut,
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
    return ImageOut(
        id=img_id,
        img="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8"),
        hist=ImageHist(r=r_intensities, g=g_intensities, b=b_intensities)
    )

