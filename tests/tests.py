#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
import json

from copy import deepcopy

thisDir = Path(__file__).parent

sys.path.insert(0, str(thisDir.parent))

from collections import OrderedDict

dict = OrderedDict

import safetensors_schema
from safetensors_schema import validate

testFile = thisDir / "tests.json"

INVALID_TENSOR_NAME = "invalid"


def testSpoiledHeaderRaises(spoilerFunc):
	def testInvalid(self):
		headerDic, invTensorHeader = self.__class__.getTemplateForInvalid()
		spoilerFunc(self, invTensorHeader)
		with self.assertRaises(ValueError):
			headerDicValidated = validate(headerDic)

	testInvalid.__name__ = spoilerFunc.__name__
	testInvalid.__doc__ = spoilerFunc.__doc__
	return testInvalid

class Tests(unittest.TestCase):
	VALID = None
	INVALID_TEMPLATE = None
	NUM_ITEM_SIZE = None
	
	@classmethod
	def setUpClass(cls):
		cls.VALID = json.loads(testFile.read_text())
		validTensorHeader =cls.VALID["I"]
		cls.INVALID_TEMPLATE = {
			INVALID_TENSOR_NAME: deepcopy(validTensorHeader)
		}
		cls.NUM_ITEM_SIZE = int(validTensorHeader["dtype"][1:]) // 8

	def testSimple(self):
		headerDic = self.__class__.VALID
		headerDicValidated = validate(headerDic)
		self.assertEqual(headerDic, headerDicValidated)

	@classmethod
	def getTemplateForInvalid(cls):
		headerDic = deepcopy(cls.INVALID_TEMPLATE)
		return headerDic, headerDic[INVALID_TENSOR_NAME]

	# must contain ('data_offsets', 'dtype', 'shape') properties
	@testSpoiledHeaderRaises
	def testNoDtype(self, invTensorHeader):
		del invTensorHeader["dtype"]

	# must contain ('data_offsets', 'dtype', 'shape') properties
	@testSpoiledHeaderRaises
	def testNoShape(self, invTensorHeader):
		del invTensorHeader["shape"]

	# must contain ('data_offsets', 'dtype', 'shape') properties
	@testSpoiledHeaderRaises
	def testNoOffsetsRange(self, invTensorHeader):
		del invTensorHeader["data_offsets"]

	# must match pattern
	@testSpoiledHeaderRaises
	def testInvalidLetter(self, invTensorHeader):
		invTensorHeader["dtype"] = "Q" + invTensorHeader["dtype"][1:]

	# must match pattern
	@testSpoiledHeaderRaises
	def testInvalidBitSize(self, invTensorHeader):
		invTensorHeader["dtype"] = invTensorHeader["dtype"][:1] + str(int(invTensorHeader["dtype"][1:]) - 1)

	#start offset of data is after the end offset of it, the size is negative
	@testSpoiledHeaderRaises
	def testReversedOffsetsRange(self, invTensorHeader):
		offsts = invTensorHeader["data_offsets"]
		invTensorHeader["data_offsets"] = type(offsts)(reversed(offsts))

	# must be natural
	@testSpoiledHeaderRaises
	def testNegativeStartOffset(self, invTensorHeader):
		invTensorHeader["data_offsets"][0] = - invTensorHeader["data_offsets"][0]

	# must be natural
	@testSpoiledHeaderRaises
	def testNegativeBothOffsets(self, invTensorHeader):
		offsts = invTensorHeader["data_offsets"]
		invTensorHeader["data_offsets"] = type(offsts)(-el for el in reversed(offsts))

	# must be of length 2
	@testSpoiledHeaderRaises
	def testNonsenseOffsetsRange0(self, invTensorHeader):
		invTensorHeader["data_offsets"] = ()

	# must be of length 2
	@testSpoiledHeaderRaises
	def testNonsenseOffsetsRange1(self, invTensorHeader):
		invTensorHeader["data_offsets"] = invTensorHeader["data_offsets"][0:1]

	# must be of length 2
	@testSpoiledHeaderRaises
	def testNonsenseOffsetsRange3(self, invTensorHeader):
		invTensorHeader["data_offsets"] = tuple(invTensorHeader["data_offsets"]) + (100500,)

	# must be equal to len(
	@testSpoiledHeaderRaises
	def testMessedDataOffsetsIncreased(self, invTensorHeader):
		invTensorHeader["data_offsets"][1] += self.__class__.NUM_ITEM_SIZE

	# must be equal to len(
	@testSpoiledHeaderRaises
	def testMessedDataOffsetsDecreased(self, invTensorHeader):
		invTensorHeader["data_offsets"][1] -= self.__class__.NUM_ITEM_SIZE

	# must be equal to len(
	@testSpoiledHeaderRaises
	def testMessedShapeIncreased(self, invTensorHeader):
		invTensorHeader["shape"][0] += 1

	# must be equal to len(
	@testSpoiledHeaderRaises
	def testMessedShapeDecreased(self, invTensorHeader):
		invTensorHeader["shape"][0] -= 1

if __name__ == "__main__":
	unittest.main()
