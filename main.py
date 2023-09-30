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
    print(xp, fp, img)
    return {
      "id": uuid4(),
      "data": "this is placeholder."
    }
