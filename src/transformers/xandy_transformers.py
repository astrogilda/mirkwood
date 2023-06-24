
from sklearn.preprocessing import StandardScaler
from pydantic import root_validator, BaseModel
from typing import List
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np
import logging

from utils.logger import LoggingUtility

logger = LoggingUtility.get_logger(
    __name__, log_file='logs/xandy_transformer.log')
logger.info('Saving logs from xandy_transformer.py')


EPS = 1e-6

# TODO: add sanity checks for transformers other than StandardScaler


class TransformerConfig(BaseModel):
    name: str
    transformer: TransformerMixin

    class Config:
        arbitrary_types_allowed: bool = True


class TransformerBase(BaseModel):
    transformers: Optional[List[TransformerConfig]]

    class Config:
        arbitrary_types_allowed: bool = True

    @root_validator(pre=True)
    def validate_transformers(cls, values):
        transformers = values.get("transformers")

        # If transformers is not None, do the validations
        if transformers is not None:
            # If transformers is not a list, raise an error
            if not isinstance(transformers, list):
                raise ValueError(
                    "transformers must be a list of TransformerConfig objects")

            # Check that each transformer in the list is an instance of TransformerConfig
            for transformer in transformers:
                if not isinstance(transformer, TransformerConfig):
                    raise ValueError(
                        "Each transformer must be an instance of TransformerConfig")

        return values


class XTransformer(TransformerBase):

    @root_validator(pre=True)
    def set_default_transformers(cls, values):
        transformers = values.get("transformers")

        # If transformers is None, set it to a list containing a default StandardScaler wrapped in a TransformerConfig
        if transformers is None:
            values["transformers"] = [TransformerConfig(
                name="standard_scaler", transformer=StandardScaler())]

        return values


class YTransformer(TransformerBase):
    pass
