import numpy as np
from fastapi import APIRouter
from fastapi import status
from fastapi import Query

from src.models.schemas.frequency_filtering import (
    FilterApplicationSchema,
    FilteringPipelineSchema,
)
from src.models.schemas.image import ImageSchema
from src.pipelines.frequency_filtering.core import (
    apply_ideal_filter,
    apply_butterworth_filter,
    apply_gaussian_filter,
)
from src.pipelines.frequency_filtering.methods import get_spectrum
from src.utils.fs_io import make_path
from src.config.base import ROOT_DIR
from src.utils.images import get_image_schema
import cv2 as cv

router = APIRouter(prefix="/frequency-filtering", tags=["frequency-filtering"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "letters.png"


@router.get(
    path="/get-spectrum",
    name="frequency-filtering:get-spectrum",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def get_spectrum_route(
    name: str = Query(description="Image name", default=default_image),
):
    img_in = cv.imread(make_path(dir_in, name), cv.IMREAD_GRAYSCALE)
    spectrum = get_spectrum(img_in)
    return get_image_schema(
        spectrum,
        "{img_name}_spectrum.png".format(img_name=name),
        include_hist=False,
    )


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


def frequency_filtering_schema_adapter(pipeline_imgs: dict):
    smoothing_schema = get_frequency_filtering_schema(
        pipeline_imgs["smoothing"]["filter"],
        pipeline_imgs["smoothing"]["spectrum"],
        pipeline_imgs["smoothing"]["img_out"],
        "smoothing",
    )
    sharpening_schema = get_frequency_filtering_schema(
        pipeline_imgs["sharpening"]["filter"],
        pipeline_imgs["sharpening"]["spectrum"],
        pipeline_imgs["sharpening"]["img_out"],
        "sharpening",
    )
    return FilteringPipelineSchema(
        smoothing_schema=smoothing_schema,
        sharpening_schema=sharpening_schema,
    )


@router.get(
    path="/apply-ideal",
    name="frequency-filtering:apply-ideal",
    response_model=FilteringPipelineSchema,
    status_code=status.HTTP_200_OK,
)
async def apply_ideal_filter_route(
    name: str = Query(description="Image name", default=default_image),
):
    img_in = cv.imread(make_path(dir_in, name), cv.IMREAD_GRAYSCALE)
    pipeline_imgs = apply_ideal_filter(img_in)
    return frequency_filtering_schema_adapter(pipeline_imgs)


@router.get(
    path="/apply-butterworth",
    name="frequency-filtering:apply-butterworth",
    response_model=FilteringPipelineSchema,
    status_code=status.HTTP_200_OK,
)
async def apply_ideal_filter_route(
    name: str = Query(description="Image name", default=default_image),
):
    img_in = cv.imread(make_path(dir_in, name), cv.IMREAD_GRAYSCALE)
    pipeline_imgs = apply_butterworth_filter(img_in)
    return frequency_filtering_schema_adapter(pipeline_imgs)


@router.get(
    path="/apply-gaussian",
    name="frequency-filtering:apply-gaussian",
    response_model=FilteringPipelineSchema,
    status_code=status.HTTP_200_OK,
)
async def apply_ideal_filter_route(
    name: str = Query(description="Image name", default=default_image),
):
    img_in = cv.imread(make_path(dir_in, name), cv.IMREAD_GRAYSCALE)
    pipeline_imgs = apply_gaussian_filter(img_in)

    return frequency_filtering_schema_adapter(pipeline_imgs)
