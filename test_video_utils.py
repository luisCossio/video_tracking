import unittest
import video_utils as vu
import numpy as np

class test_utils(unittest.TestCase):
    def setUp(self):
        pass

    def test_cut_functions1(self):
        shape_img = [500, 360]
        dummy_img = np.random.randint(0,255,shape_img+[3])

        shape_cuts = [50,36]
        split_img = [10,10]

        cuts_img,cut_locations = vu.get_cuts(dummy_img,shape_cuts, splits=split_img)

        expected_locations = np.empty([split_img[0],split_img[1],2])
        expected_locations[:, :, 0] = np.repeat(np.arange(split_img[1]).reshape([-1,1])*shape_cuts[0], split_img[1],
                                                axis=1)

        expected_locations[:, :, 1] = np.repeat(np.arange(split_img[1]).reshape([1,-1])*shape_cuts[1], split_img[0],
                                                axis=0)

        cut_locations = np.array(cut_locations,dtype=np.int32)

        for i in range(split_img[0]):
            for j in range(split_img[1]):
                self.assertEqual(cut_locations[i,j,0],expected_locations[i,j,0])
                self.assertEqual(cut_locations[i,j,1],expected_locations[i,j,1])

                pixel_img = dummy_img[cut_locations[i,j,0],cut_locations[i,j,1],:]
                pixel_cut = cuts_img[i][j][0,0,:]
                np.testing.assert_array_equal(pixel_img, pixel_cut)


    def test_cut_functions2(self):
        """
        Test 2: Verifying that we get evenly distribbuted patches, even if they do not cover the entire patch
        :return:
        """
        shape_img = [900, 900]
        dummy_img = np.random.randint(0,255,shape_img+[3])

        shape_cuts = [150,200]
        split_img = [2,3]

        cuts_img, cut_locations = vu.get_cuts(dummy_img,shape_cuts, splits=split_img)

        expected_locations = np.array([[[0,0], [0,350], [0,700]],
                                       [[750,0], [750,350], [750,700]],
                                       ])

        cut_locations = np.array(cut_locations,dtype=np.int32)

        for i in range(split_img[0]):
            for j in range(split_img[1]):
                self.assertEqual(cut_locations[i,j,0],expected_locations[i,j,0])
                self.assertEqual(cut_locations[i,j,1],expected_locations[i,j,1])

                pixel_img = dummy_img[cut_locations[i,j,0],cut_locations[i,j,1],:]
                pixel_cut = cuts_img[i][j][0,0,:]
                np.testing.assert_array_equal(pixel_img, pixel_cut)



