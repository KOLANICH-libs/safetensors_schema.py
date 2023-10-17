from warnings import warn

warn("We have moved from M$ GitHub to https://codeberg.org/KFmts/safetensors_schema.py , read why on https://codeberg.org/KOLANICH/Fuck-GuanTEEnomo .")

# generated with https://github.com/horejsek/python-fastjsonschema with some manual postprocessing

__all__ = ("validate",)

from functools import reduce
from operator import mul

import re

typePattern = re.compile("([UIF])(8|16|32|64|128|256)")

METADATA_PROP_NAME = "__metadata__"
tensorPropKeys = ("data_offsets", "dtype", "shape")


def validate(data, custom_formats={}, name_prefix=None):
	if name_prefix is None:
		name_prefix = "data"

	if not isinstance(data, dict):
		raise ValueError(name_prefix + " must be object", data, name_prefix, "type")

	data_len = len(data)
	if False and not all(prop in data for prop in [METADATA_PROP_NAME]):  # branch intentionally omitted, __metadata__ can be missing for now
		raise ValueError(name_prefix + " must contain ['" + METADATA_PROP_NAME + "'] properties", data, name_prefix, "required")
	tensorNames = set(data.keys())
	if METADATA_PROP_NAME in tensorNames:
		tensorNames.remove(METADATA_PROP_NAME)
		validate_metadata(data[METADATA_PROP_NAME], custom_formats, name_prefix + "." + METADATA_PROP_NAME)

	for tensor_name in tensorNames:
		data_value = data.get(tensor_name)
		validate_tensor(data_value, custom_formats, name_prefix + "." + tensor_name)
	return data

sizeTLimit = ((1 << 48) - 1)

def validate_dtype(dtype, name_prefix=None):
	if name_prefix is None:
		name_prefix = "dtype"

	if not isinstance(dtype, str):
		raise ValueError(name_prefix + ".dtype must be string", dtype, name_prefix + ".dtype", {"type": "string", "pattern": typePattern.pattern}, "type")
	typePattern_match = typePattern.match(dtype)
	if not typePattern_match:
		raise ValueError(name_prefix + ".dtype must match pattern " + typePattern.pattern, dtype, name_prefix + ".dtype", {"type": "string", "pattern": typePattern.pattern}, "pattern")
	dtype_category, dtype_size = typePattern_match.groups()
	dtype_bitSize = int(dtype_size)
	excessBits = dtype_bitSize % 8
	if excessBits:
		raise ValueError(name_prefix + ".dtype must be integer count of bytes, but there are excess bits", excessBits, dtype_bitSize)
	dtype_byteSize = dtype_bitSize // 8
	return dtype, dtype_byteSize

def validate_size_t(size_t_num, name_prefix=None):
	if name_prefix is None:
		name_prefix = "size_t_num"

	if not isinstance(size_t_num, int):
		raise ValueError(name_prefix  + " must be integer", size_t_num, name_prefix, {"type": "integer"}, "type")
	if size_t_num < 0:
		raise ValueError(name_prefix + " must be natural", size_t_num, name_prefix, {"minimum": 0}, "minimum")
	if size_t_num > sizeTLimit:
		raise ValueError(name_prefix + " must be CPU-addressible", size_t_num, name_prefix, {"maximum": sizeTLimit}, "maximum")
	return size_t_num

def validate_shape(shape, name_prefix=None):
	if name_prefix is None:
		name_prefix = "shape"
	
	if not isinstance(shape, (list, tuple)):
		raise ValueError(name_prefix + " must be array", shape,  {"type": "array", "items": {"type": "integer"}}, "type")

	for i, shape_item in enumerate(shape):
		validate_size_t(shape_item, name_prefix + ".shape[" + str(i) + "]")

	items_count = validate_size_t(product(shape), "product("  + name_prefix + ")")

	return shape, items_count

def product(els):
	return reduce(mul, els, 1)

def validate_data_offsets(offsets, dtype_byteSize, name_prefix=None):
	if not isinstance(offsets, (list, tuple)):
		raise ValueError(name_prefix + " must be array", offsets, name_prefix + "", {"type": "array", "items": {"type": "integer"}}, "type")
	if not len(offsets) == 2:
		raise ValueError(name_prefix + "  must be of length 2", {"type": "array", "prefixItems": [..., ...]}, "type")
	for i, offsets_item in enumerate(offsets):
		validate_size_t(offsets_item, name_prefix + "[" + str(i) + "]" )

	size = offsets[1] - offsets[0]
	if size < 0:
		raise ValueError("start offset of data is after the end offset of it, the size is negative")
	excessBytes = size % dtype_byteSize
	if excessBytes:
		raise ValueError(name_prefix + " must specify the range that has no bytes belonging not to a value, but there are excess bytes", excessBytes, size)
	return offsets, size

def validate_tensor(data, custom_formats={}, name_prefix=None):
	if name_prefix is None:
		name_prefix = "data"
	
	if not isinstance(data, dict):
		raise ValueError(name_prefix + " must be object", data, name_prefix, "type")

	data_len = len(data)
	if not all(prop in data for prop in tensorPropKeys):
		raise ValueError(name_prefix + " must contain " + repr(tensorPropKeys) + " properties", data, name_prefix, "required")
	ks = set(data.keys()) - {"dtype", "shape", "data_offsets"}
	
	dtype, dtype_byteSize = validate_dtype(data["dtype"], name_prefix)
	shape, items_count = validate_shape(data["shape"], name_prefix + ".shape")
	
	dataOffsetsFormula = name_prefix + ".data_offsets"
	offsets, size = validate_data_offsets(data["data_offsets"], dtype_byteSize, dataOffsetsFormula)
	
	computedSizeFormula = "product(" + name_prefix + ".shape) * " + name_prefix + ".data_offsets"
	computedSize = validate_size_t(items_count * dtype_byteSize, computedSizeFormula)
	if computedSize != size:
		raise ValueError(computedSizeFormula +  " must be equal to len(" + dataOffsetsFormula + ")")

	if ks:
		raise ValueError(name_prefix + " must not contain " + str(ks) + " properties", data, name_prefix, "additionalProperties")
	return data


def validate_metadata(data, custom_formats={}, name_prefix=None):
	if name_prefix is None:
		name_prefix = "data"

	if not isinstance(data, dict):
		raise ValueError(name_prefix + " must be object", data, name_prefix, {"type": "object", "additionalProperties": {"type": "string"}, "title": "Metadata"}, "type")

	ks = set(data.keys())
	for k in ks:
		data_value = data.get(k)
		if not isinstance(data_value, str):
			raise ValueError(name_prefix + "." + k + " must be string", data_value, name_prefix + "." + k, {"type": "string"}, "type")
	return data
