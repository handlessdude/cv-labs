from fastapi import APIRouter
from uuid import uuid4
from fastapi import status
from fastapi import Query
from loguru import logger
import numpy as np
from src.pipelines.image_description.methods import describe_channels
from src.pipelines.quantization.methods import (
    convert_to_halftone,
    convert_to_quantitized,
    otsu_global_binarization,
)
from src.pipelines.quantization.otsu import otsu_local_binarization
from src.models.schemas.image import ImageSchema, ImageHist
from src.utils.fs_io import open_img
from src.config.base import ROOT_DIR
from src.utils.fs_io import img_to_base64

router = APIRouter(prefix="/quantization", tags=["quantization"])

dir_in = f"{str(ROOT_DIR)}/app-data/inputs"
default_image = "gosling1.png"


def get_image_schema(img_in: np.ndarray, name: str):
    ri_out, gi_out, bi_out = describe_channels(img_in)
    return ImageSchema(
        id=uuid4(),
        name=name,
        img_src=img_to_base64(img_in),
        hist=ImageHist(r=ri_out, g=gi_out, b=bi_out),
    )


@router.get(
    path="/halftone",
    name="quantization:halftone",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def halftone(
    name: str = Query(description="Image name"),
):
    logger.info("Processing halftone for image {name}", name=name)
    img_in = open_img(dir_in, name)
    img_out = convert_to_halftone(img_in)
    img_out_schema = get_image_schema(img_out, name)
    logger.info("Processing image {name} done", name=name)
    return img_out_schema


@router.get(
    path="/quantitize",
    name="quantization:quantitize",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def quantitize(
    name: str = Query(description="Image name"),
    levels: str = Query(description="List of quantization levels"),
):
    logger.info("Processing quantization for image {name}", name=name)
    levels = [int(x) for x in levels.split(",")]
    img_in = open_img(dir_in, name)
    img_out = convert_to_quantitized(img_in, levels)
    img_out_schema = get_image_schema(img_out, name)
    logger.info("Processing image {name} done", name=name)
    return img_out_schema


@router.get(
    path="/otsu-global",
    name="quantization:otsu-global",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def otsu_global(
    name: str = Query(description="Image name"),
):
    img_in_id = uuid4()
    logger.info("Processing quantization for image {img_id}", img_id=img_in_id)
    img_in = open_img(dir_in, name)
    img_out = otsu_global_binarization(img_in)
    img_out_schema = get_image_schema(img_out, name)
    logger.info("Processing image {img_id} done", img_id=img_in_id)
    return img_out_schema


@router.get(
    path="/otsu-local",
    name="quantization:otsu-local",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def otsu_local(
    name: str = Query(description="Image name"),
):
    img_in_id = uuid4()
    logger.info("Processing quantization for image {img_id}", img_id=img_in_id)
    img_in = open_img(dir_in, name)
    img_out = otsu_local_binarization(img_in)
    img_out_schema = get_image_schema(img_out, name)
    logger.info("Processing image {img_id} done", img_id=img_in_id)
    return img_out_schema


@router.get(
    path="/otsu-hierarchical",
    name="quantization:otsu-hierarchical",
    response_model=ImageSchema,
    status_code=status.HTTP_200_OK,
)
async def otsu_hierarchical(
    name: str = Query(description="Image name"),
):
    img_in_id = uuid4()
    logger.info("Processing quantization for image {img_id}", img_id=img_in_id)
    img_in = open_img(dir_in, name)
    img_out = otsu_global_binarization(img_in)
    img_out_schema = get_image_schema(img_out, name)
    logger.info("Processing image {img_id} done", img_id=img_in_id)
    return img_out_schema
