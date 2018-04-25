from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Interface class
class Conv(object):
	def __init__(self, shape, name=None):
		self.shape = shape
		self.name = name

	def conv(self):
		return self.graph

# Conv 3 1D
class ConvThree1D(Conv):
	def __init__(self, shape, name=None, init1=None, init2=None, init3=None):
		super(ConvThree1D, self).__init__(shape, name)

class ConvOne1DOne2D(Conv):
	def __init__(self, shape, name=None, init1=None, init2=None):
		super(ConvThree1D, self).__init__(shape, name)

class ConvOne2DOne1D(Conv):
	def __init__(self, shape, name=None, init1=None, init2=None):
		super(ConvThree1D, self).__init__(shape, name)

class Conv3D(Conv):
	def __init__(self, shape, name=None, init1=None):
		super(ConvThree1D, self).__init__(shape, name)