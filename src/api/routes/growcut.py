from fastapi import APIRouter
from fastapi import status
from fastapi import Query

from src.models.schemas.image import ImageSchema

from src.utils.fs_io import make_path
from src.config.base import ROOT_DIR
from src.utils.images import get_image_schema
import cv2 as cv

router = APIRouter(prefix="/growcut", tags=["growcut"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "gosling0.png"


@router.get(
    path="/cut-foreground",
    name="growcut:cut-foreground",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def cut_foreground_route(
    name: str = Query(description="Image name", default=default_image),
    # some foreground/background bullshit drawings
):
    img_in = cv.imread(make_path(dir_in, name), cv.COLOR_BGR2RGB)
    return get_image_schema(
        img_in,
        name,
        include_hist=False,
    )
