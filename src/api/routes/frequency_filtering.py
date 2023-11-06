from fastapi import APIRouter
from fastapi import status
from fastapi import Query
from src.models.schemas.image import ImageSchema
from src.pipelines.frequency_filtering.core import fft_smoothen_sharpen
from src.utils.fs_io import open_img, make_path
from src.config.base import ROOT_DIR
from src.models.schemas.base import BaseSchemaModel
from src.utils.images import get_image_schema
import cv2 as cv

router = APIRouter(prefix="/frequency-filtering", tags=["frequency-filtering"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "letters.png"


class Test(BaseSchemaModel):
    source: ImageSchema
    spectrum: ImageSchema


@router.get(
    path="/smoothening-sharpening",
    name="frequency-filtering:smoothening-sharpening",
    response_model=Test,
    status_code=status.HTTP_200_OK,
)
async def smoothen_sharpen(
    name: str = Query(description="Image name", default=default_image),
):
    # img_in = cv.imread(make_path(dir_in, default_image), cv.IMREAD_GRAYSCALE)
    img_in = cv.cvtColor(
        cv.imread(make_path(dir_in, default_image), cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY
    )

    pipeline_imgs = fft_smoothen_sharpen(img_in)
    spectrum = get_image_schema(
        pipeline_imgs["spectrum"],
        "spectrum.png",
        include_hist=False,
    )
    return Test(
        source=get_image_schema(img_in, name, include_hist=False),
        spectrum=spectrum,
    )
