from rectpack import newPacker
import math


class RECTANGLE_PACKER:
	"""
	Documentation of rectpack:
	https://github.com/secnot/rectpack/
	"""

	def __init__(self, rectangles, rectangle_names, bin):
		"""

		:param rectangles: list of (width, height) dimensions
		:param rectangle_names: list of names of the rectangles
		:param height_limit: fixed height of the container
		"""
		self.packer = newPacker(rotation=True)
		# Add the rectangles to packing queue
		for i, r in enumerate(rectangles):
			self.packer.add_rect(*r, rid=rectangle_names[i])
		# Add the bins where the rectangles will be placed
		self.packer.add_bin(*bin)

		list_dict = []
		for rectangle in rectangles:
			list_dict.append({'width': rectangle[0], 'height': rectangle[1], 'rotable': True})
		self.rectangle_names = rectangle_names
		self.rectangles = rectangles

	def solve(self, show_result=False):
		"""

		:return: dictionary with key, value pairs of the form rid: (x, y, w, h, b) where rid is a rectangle id,
				x = rectangle bottom-left x coordinate
				y = rectangle bottom-left y coordinate
				w = rectangle width
				h = rectangle height
				b = bin index
		"""
		# Pack using rectPack
		self.packer.pack()
		# Obtain number of bins used for packing
		nbins = len(self.packer)
		# if no bin is used
		if nbins == 0:
			# no solution is found
			return None
		# otherwise
		else:
			# Number of rectangles packed into first bin
			num_rect_packed = len(self.packer[0])
			# Number of rectangles to pack
			num_rect_to_pack = len(self.packer._avail_rect)
			# if all the rectangles could not be packed in the first bin
			if num_rect_packed < num_rect_to_pack:
				# no solution is found
				return None
			# a solution has been found
			else:
				# Initialize dictionary for the solution
				arrangement_dict = {}
				required_width = 0
				required_height = 0
				# Full rectangle list
				all_rects = self.packer.rect_list()
				for rect in all_rects:
					b, x, y, w, h, rid = rect
					arrangement_dict.update({rid: (x, y, w, h, b)})
					# b - Bin index
					# x - Rectangle bottom-left corner x coordinate
					# y - Rectangle bottom-left corner y coordinate
					# w - Rectangle width
					# h - Rectangle height
					# rid - User assigned rectangle id or None
					if x + w > required_width:
						required_width = x + w
					if y + h > required_height:
						required_height = y + h
					if show_result:
						print('Place bottom left corner of rectangle #' + str(rid) + ' of size W' + str(w) + 'xH' + str(h) + ' at (' + str(x) + ',' + str(y) + ') in bin ' + str(b))
				print('RectPack: Required height = ' + str(required_height) + ', required length = ' + str(required_width))
				return arrangement_dict




