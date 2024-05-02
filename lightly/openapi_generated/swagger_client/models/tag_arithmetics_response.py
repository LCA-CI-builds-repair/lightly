# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
from inspect import getfullargspec
import json
import pprint
import re  # noqa: F401

from typing import Any, List, Optional
from pydantic import BaseModel, Field, StrictStr, ValidationError, validator
from lightly.openapi_generated.swagger_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_generated.swagger_client.models.tag_bit_mask_response import TagBitMaskResponse
from typing import Any, List
from pydantic import StrictStr, Field, Extra

TAGARITHMETICSRESPONSE_ONE_OF_SCHEMAS = ["CreateEntityResponse", "TagBitMaskResponse"]

class TagArithmeticsResponse(BaseModel):
    """
    TagArithmeticsResponse
    """
    # data type: CreateEntityResponse
    oneof_schema_1_validator: Optional[CreateEntityResponse] = None
    # data type: TagBitMaskResponse
    oneof_schema_2_validator: Optional[TagBitMaskResponse] = None
    actual_instance: Any
    one_of_schemas: List[str] = Field(TAGARITHMETICSRESPONSE_ONE_OF_SCHEMAS, const=True)

    class Config:
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def __init__(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise ValueError("If a position argument is used, only 1 is allowed to set `actual_instance`")
            if kwargs:
                raise ValueError("If a position argument is used, keyword arguments cannot be used.")
            super().__init__(actual_instance=args[0])
        else:
            super().__init__(**kwargs)

    @validator('actual_instance')
    def actual_instance_must_validate_oneof(cls, v):
        instance = TagArithmeticsResponse.construct()
        error_messages = []
        match = 0
        # validate data type: CreateEntityResponse
        if not isinstance(v, CreateEntityResponse):
            error_messages.append(f"Error! Input type `{type(v)}` is not `CreateEntityResponse`")
        match = 0
        # validate data type: TagBitMaskResponse
        if not isinstance(v, TagBitMaskResponse):
            error_messages.append(f"Error! Input type `{type(v)}` is not `TagBitMaskResponse`")
            match += 1
        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when setting `actual_instance` in TagArithmeticsResponse with oneOf schemas: CreateEntityResponse, TagBitMaskResponse. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when setting `actual_instance` in TagArithmeticsResponse with oneOf schemas: CreateEntityResponse, TagBitMaskResponse. Details: " + ", ".join(error_messages))
        else:
            return v

    @classmethod
    def from_dict(cls, obj: dict) -> TagArithmeticsResponse:
        return cls.from_json(json.dumps(obj))

    @classmethod
    def from_json(cls, json_str: str) -> TagArithmeticsResponse:
        """Returns the object represented by the json string"""
        instance = TagArithmeticsResponse.construct()
        error_messages = []
        match = 0

        # deserialize data into CreateEntityResponse
        try:
            instance.actual_instance = CreateEntityResponse.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into TagBitMaskResponse
        try:
            instance.actual_instance = TagBitMaskResponse.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))

        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when deserializing the JSON string into TagArithmeticsResponse with oneOf schemas: CreateEntityResponse, TagBitMaskResponse. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when deserializing the JSON string into TagArithmeticsResponse with oneOf schemas: CreateEntityResponse, TagBitMaskResponse. Details: " + ", ".join(error_messages))
        else:
            return instance

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the actual instance"""
        if self.actual_instance is None:
            return "null"

        to_json = getattr(self.actual_instance, "to_json", None)
        if callable(to_json):
            return self.actual_instance.to_json(by_alias=by_alias)
        else:
            return json.dumps(self.actual_instance)

    def to_dict(self, by_alias: bool = False) -> dict:
        """Returns the dict representation of the actual instance"""
        if self.actual_instance is None:
            return None

        to_dict = getattr(self.actual_instance, "to_dict", None)
        if callable(to_dict):
            return self.actual_instance.to_dict(by_alias=by_alias)
        else:
            # primitive type
            return self.actual_instance

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the actual instance"""
        return pprint.pformat(self.dict(by_alias=by_alias))

