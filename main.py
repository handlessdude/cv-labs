from typing import List
from pydantic import BaseModel
from pydantic import UUID4
from uuid import uuid4
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import ValidationException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from loguru import logger

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


class CorrectionOut(BaseModel):
    id: UUID4
    data: str


@app.get(
    path="/color-correction/spline",
    name="color-correction:spline",
    response_model=CorrectionOut,
    status_code=status.HTTP_200_OK,
)
async def color_correction_spline(
  img: str = Query(description="Image source in base64 format"),
  xp: List[float] = Query(description="Array of x coordinates of interpolation points"),
  fp: List[float] = Query(description="Array of y coordinates of interpolation points"),
):
    img_id = uuid4()
    logger.info("Processing image {img_id} with spline {xp}, {fp}", img_id=img_id, xp=xp, fp=fp)
    logger.info("Processing image {img_id} done", img_id=img_id)
    return {
      "id": img_id,
      "data": "this is placeholder."
    }
