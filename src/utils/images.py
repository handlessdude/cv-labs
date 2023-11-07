from uuid import uuid4

import numpy as np

from src.models.schemas.image import ImageSchema, ImageHist
from src.pipelines.image_description.methods import describe_channels
from src.utils.fs_io import img_to_base64


## todo refactor
def get_image_schema(img_in: np.ndarray, name: str, include_hist: bool = True):
    if include_hist:
        ri_out, gi_out, bi_out = describe_channels(img_in)
        return ImageSchema(
            id=uuid4(),
            name=name,
            img_src=img_to_base64(img_in),
            hist=ImageHist(r=ri_out, g=gi_out, b=bi_out) if include_hist else None,
        )
    return ImageSchema(
        id=uuid4(),
        name=name,
        img_src=img_to_base64(img_in),
    )
