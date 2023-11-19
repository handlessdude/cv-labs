from fastapi import APIRouter
from fastapi import status
from pydantic import BaseModel


import uuid

from src.pipelines.growcut.core import growcut
from src.utils.fs_io import base64_to_ndarray, save_img
from src.config.base import ROOT_DIR
from src.utils.images import get_image_schema


router = APIRouter(prefix="/growcut", tags=["growcut"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
dir_out = f"{str(ROOT_DIR)}/app-data/outputs/growcut"
default_image = "gosling0.png"


class GrowcutData(BaseModel):
    src_img: str
    markings: str


@router.post(
    path="/cut-object",
    name="growcut:cut-object",
    # response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def cut_foreground_route(data: GrowcutData):
    img_in = base64_to_ndarray(data.src_img)
    markings = base64_to_ndarray(data.markings)

    img_out = growcut(img_in, markings)

    item_dir_out = dir_out + "/" + str(uuid.uuid4())
    save_img(img_in, item_dir_out, "input.png")
    save_img(markings, item_dir_out, "markings.png")
    save_img(img_out, item_dir_out, "output.png")

    return get_image_schema(
        img_out,
        "growcut.png",
        include_hist=False,
    )
