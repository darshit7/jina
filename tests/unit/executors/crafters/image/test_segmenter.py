import numpy as np
from jina.executors.crafters.image.segmenter import RandomImageCropper, FiveImageCropper, \
    SlidingWindowImageCropper
from tests.unit.executors.crafters.image import JinaImageTestCase


class ImageSegmentTestCase(JinaImageTestCase):
    def test_random_crop(self):
        img_size = 217
        img_array = self.create_random_img_array(img_size, img_size)
        output_dim = 20
        num_pathes = 20
        crafter = RandomImageCropper(output_dim, num_pathes)
        crafted_chunk_list = crafter.craft(img_array, 0, 0)
        self.assertEqual(len(crafted_chunk_list), num_pathes)

    def test_five_image_crop(self):
        img_size = 217
        img_array = self.create_random_img_array(img_size, img_size)
        output_dim = 20
        crafter = FiveImageCropper(output_dim)
        crafted_chunk_list = crafter.craft(img_array, 0, 0)
        self.assertEqual(len(crafted_chunk_list), 5)

    def test_sliding_windows_no_padding(self):
        img_size = 14
        img_array = self.create_random_img_array(img_size, img_size)
        output_dim = 4
        strides = (6, 6)
        crafter = SlidingWindowImageCropper(target_size=output_dim, strides=strides, padding=False)
        chunks = crafter.craft(img_array, 0, 0)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]['location'], (0, 0))
        self.assertEqual(chunks[1]['location'], (0, 6))
        self.assertEqual(chunks[2]['location'], (6, 0))
        self.assertEqual(chunks[3]['location'], (6, 6))

    def test_sliding_windows_with_padding(self):
        img_size = 14
        img_array = self.create_random_img_array(img_size, img_size)
        output_dim = 4
        strides = (6, 6)
        crafter = SlidingWindowImageCropper(target_size=output_dim, strides=strides, padding=True)
        chunks = crafter.craft(img_array, 0, 0)
        self.assertEqual(len(chunks), 9)
        self.assertEqual(chunks[0]['location'], (0, 0))
        self.assertEqual(chunks[1]['location'], (0, 6))
        self.assertEqual(chunks[2]['location'], (0, 12))
        self.assertEqual(chunks[3]['location'], (6, 0))
        self.assertEqual(chunks[4]['location'], (6, 6))
        self.assertEqual(chunks[5]['location'], (6, 12))
        self.assertEqual(chunks[6]['location'], (12, 0))
        self.assertEqual(chunks[7]['location'], (12, 6))
        self.assertEqual(chunks[8]['location'], (12, 12))
