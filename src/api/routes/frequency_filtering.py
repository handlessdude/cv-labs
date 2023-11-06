import numpy as np
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
default_image = "flower.jpg"


class FilterApplicationSchema(BaseSchemaModel):
    filt: ImageSchema
    spectrum: ImageSchema
    img_out: ImageSchema


class FrequencyFilteringSchema(BaseSchemaModel):
    source: ImageSchema
    spectrum: ImageSchema
    smoothing_schema: FilterApplicationSchema


def get_frequency_filtering_schema(
    filt: np.ndarray, spectrum: np.ndarray, img_out: np.ndarray, schema_name: str
):
    return FilterApplicationSchema(
        filt=get_image_schema(
            filt,
            "{schema_name}_filter.png".format(schema_name=schema_name),
            include_hist=False,
        ),
        spectrum=get_image_schema(
            spectrum,
            "{schema_name}_spectrum.png".format(schema_name=schema_name),
            include_hist=False,
        ),
        img_out=get_image_schema(
            img_out,
            "{schema_name}_img_out.png".format(schema_name=schema_name),
            include_hist=False,
        ),
    )


@router.get(
    path="/smoothening-sharpening",
    name="frequency-filtering:smoothening-sharpening",
    response_model=FrequencyFilteringSchema,
    status_code=status.HTTP_200_OK,
)
async def smoothen_sharpen(
    name: str = Query(description="Image name", default=default_image),
):
    # img_in = cv.cvtColor(
    #     cv.imread(make_path(dir_in, default_image), cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY
    # )
    img_in = cv.imread(make_path(dir_in, default_image), cv.IMREAD_GRAYSCALE)

    source_schema = get_image_schema(img_in, name, include_hist=False)

    # print(img_in.shape) # h, w, channels
    pipeline_imgs = fft_smoothen_sharpen(img_in)

    spectrum_schema = get_image_schema(
        pipeline_imgs["spectrum"],
        "spectrum.png",
        include_hist=False,
    )
    smoothing_schema = get_frequency_filtering_schema(
        pipeline_imgs["smoothing"]["filter"],
        pipeline_imgs["smoothing"]["spectrum"],
        pipeline_imgs["smoothing"]["img_out"],
        "smoothing",
    )
    return FrequencyFilteringSchema(
        source=source_schema,
        spectrum=spectrum_schema,
        smoothing_schema=smoothing_schema,
    )
