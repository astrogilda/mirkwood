from pydantic import parse_obj_as
from sklearn.base import TransformerMixin
from ngboost import NGBRegressor


class TransformerMixinField(TransformerMixin):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if value is not None and not isinstance(value, TransformerMixin):
            raise ValueError('Value must be an instance of TransformerMixin')
        return parse_obj_as(TransformerMixin, value)


class NGBRegressorField(NGBRegressor):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if value is not None and not isinstance(value, NGBRegressor):
            raise ValueError('Value must be an instance of NGBRegressor')
        return parse_obj_as(NGBRegressor, value)
