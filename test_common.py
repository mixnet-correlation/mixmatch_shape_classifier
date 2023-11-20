import unittest
import numpy as np
import numpy.testing as test
from tensorflow.experimental.numpy import moveaxis

from common import *


class TestArrayManipulationMethods(unittest.TestCase):
    def test_rolling_window(self):
        # step < wsize
        a = np.arange(3 * 10).reshape((3, 10))
        b = rolling_window(a, 4, 2)
        self.assertEqual(b.shape, (3, 3, 4))
        
        # step == wsize
        a = np.arange(3 * 10).reshape((3, 10))
        b = rolling_window(a, 4, 4)
        self.assertEqual(b.shape, (3, 2, 4))
        
        # 3 dimensions
        a = np.arange(3 * 10 * 2).reshape((3, 10, 2))
        b = rolling_window(a, 4, 4)
        self.assertEqual(b.shape, (3, 2, 4, 2))
        
        # concrete array no overlap
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])
        expected = np.array([[[0, 1], [2, 3]],
                             [[5, 6], [7, 8]]])
        result = rolling_window(a, 2, 2)
        test.assert_array_equal(result, expected)
        
        # concrete array with overlap
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])
        expected = np.array([[[0, 1], [1, 2], [2, 3], [3, 4]],
                             [[5, 6], [6, 7], [7, 8], [8, 9]]])
        result = rolling_window(a, 2, 1)
        test.assert_array_equal(result, expected)
    
        
    def test_windowize(self):
        # addn = 1
        a = np.arange(3 * 10).reshape((3, 10))
        result = windowize(a, 4, 1, axis=1)
        expected = rolling_window(a, 4, 4, axis=1)
        test.assert_array_equal(result, expected)
        
        # addn = 0.5
        a = np.arange(3 * 10).reshape((3, 10))
        result = windowize(a, 4, 0.5, axis=1)
        expected = rolling_window(a, 4, 2, axis=1)
        test.assert_array_equal(result, expected)        
        
    
    def test_repeat(self):
        # concrete array
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])
        
        expected = np.array([[0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4],
                             
                             [5, 6, 7, 8, 9],
                             [5, 6, 7, 8, 9]])
        result = repeat(a, a.shape[0])
        test.assert_array_equal(result, expected)
        
        # 3 dimensions
        a = np.arange(3 * 10 * 2).reshape((3, 10, 2))
        b = repeat(a, a.shape[0])
        self.assertEqual(b.shape, (9,  10, 2))
        test.assert_array_equal(b[0, :, :], b[1, :, :])

    
    def test_tile(self):
        # concrete array
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])
        
        expected = np.array([[0, 1, 2, 3, 4],
                             [5, 6, 7, 8, 9],
                             
                             [0, 1, 2, 3, 4],
                             [5, 6, 7, 8, 9]])
        result = tile(a, a.shape[0])
        test.assert_array_equal(result, expected)
        
        # 3 dimensions
        a = np.arange(3 * 10 * 2).reshape((3, 10, 2))
        b = tile(a, a.shape[0])
        self.assertEqual(b.shape, (9, 10, 2))
        test.assert_array_equal(b[0, :, :], b[3, :, :])

    def test_get_pos_indices(self):
        nwins = 3
        nflows = 5
        res = get_pos_indices(nwins, nflows)
        pos = np.array([ 0,  1,  2,
                         18, 19, 20,
                         36, 37, 38,
                         54, 55, 56,
                         72, 73, 74])
        test.assert_array_equal(res, pos)
        
    def test_flatten_windows(self):
        a = np.arange(100).reshape((5, 2, 10))    
        res = flatten_windows(a)
        exp = np.arange(100).reshape((10, 10))
        test.assert_array_equal(res, exp)
        
    def test_window_view(self):
        num_wins = 2
        a = np.arange(100).reshape((5, 2, 10))
        test.assert_array_equal(a, window_view(flatten_windows(a), num_wins))    
    
    def test_embedding_reshape(self):
        nflows = 2
        nwins = 3
        size = 4
        
        embs = np.array([
         # first init
         ## i1-r1 pair
         [1, 2, 3, 7],    ### win1
         [4, 5, 6, 7],    ### win2
         [4, 5, 6, 7],    ### win3
           
         ## i1-r2 pair  
         [4, 5, 6, 7],    ### win1
         [4, 5, 6, 7],    ### win2
         [4, 5, 6, 7],    ### win3
           
         # second init  
         ## i2-r1  
         [1, 2, 3, 7],    ### win1
         [4, 5, 6, 7],    ### win2
         [4, 5, 6, 7],    ### win3
           
         ## i2-r2  
         [4, 5, 6, 7],    ### win1
         [4, 5, 6, 7],    ### win2
         [4, 5, 6, 7]     ### win3  
         ])
        self.assertEqual(embs.shape, (nwins * nflows * nflows, size))
        new = np.array([
         # first init
         ## i1-r1 pair
         [[[1, 2, 3, 7],   ### win1
          [4, 5, 6, 7],    ### win2
          [4, 5, 6, 7]],   ### win3
         
         ## i1-r2 pair
         [[4, 5, 6, 7],    ### win1
          [4, 5, 6, 7],    ### win2
          [4, 5, 6, 7]]],  ### win3
         
         # second init
         ## i2-r1
         [[[1, 2, 3, 7],   ### win1
          [4, 5, 6, 7],    ### win2
          [4, 5, 6, 7]],   ### win3
         
         ## i2-r2
         [[4, 5, 6, 7],    ### win1
          [4, 5, 6, 7],    ### win2
          [4, 5, 6, 7]]]   ### win3  
         ])
        self.assertEqual(new.shape, (nflows, nflows, nwins, size))
        test.assert_array_equal(embs[3:6, :], new[0, 1, :, :])
        
        res = embs.reshape((nflows, nflows, nwins, size))
        test.assert_array_equal(new, res)

    def test_join(self):
        a = np.array([[2, 5, 6,      8, np.nan],
                      [3, 7, 8,      8,     10]])
        b = np.array([[3, 4, 5, np.nan, np.nan],
                      [4, 5, 9,     10, np.nan]])
        c = join(a, b)
        exp = np.array([[2, 3, 4, 5, 5, 6, 8, np.nan, np.nan, np.nan],
                        [3, 4, 5, 7, 8,  8, 9, 10, 10, np.nan]])   
        test.assert_array_equal(c, exp)
    
    def test_concatenate_pairwise(self):
        a = np.array([[2, 5, 6,      8, np.nan],
                      [3, 7, 8,      8,     10]])
        b = np.array([[3, 4, 5, np.nan, np.nan],
                      [4, 5, 9,     10, np.nan]])
        c = concatenate_pairwise(a, b)
        exp = np.array([[2, 3, 4, 5, 5, 6, 8, np.nan, np.nan, np.nan],
                        [2, 4, 5, 5, 6, 8, 9, 10, np.nan, np.nan],
                        [3, 3, 4, 5, 7, 8, 8, 10, np.nan, np.nan],
                        [3, 4, 5, 7, 8,  8, 9, 10, 10, np.nan],])   
        test.assert_array_equal(c, exp)        
        
        # different number of rows
        a = np.array([[2, 5, 6,      8, np.nan],])
        b = np.array([[3, 4, 5, np.nan, np.nan],
                      [4, 5, 9,     10, np.nan]])
        c = concatenate_pairwise(a, b)
        exp = np.array([[2, 3, 4, 5, 5, 6, 8, np.nan, np.nan, np.nan],
                        [2, 4, 5, 5, 6, 8, 9, 10, np.nan, np.nan]])   
        test.assert_array_equal(c, exp)
        
    def test_calculate_similarities(self):
        # similarity matrix
        a = np.array([[1, 0, 0, 0],
                      [1, 0, 0, 0],
                      
                      [0, 1, 0, 0],
                      [0, 1, 0, 0],
                      
                      
                      [0, 0, 1, 0],
                      [0, 0, 1, 0],
                      
                      [0, 0, 1, 0],
                      [0, 0, 1, 0]]).astype(float)
        
        b = np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      
                      
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                      
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]).astype(float)

        sims = cosine_similarity(convert_to_tensor(a, add_axis=False),
                                 convert_to_tensor(b, add_axis=False))
        sims = tf.reshape(sims, shape=(2, 2, 2))
        
        exp = np.array([[[0, 1], [0, 0]], [[0, 0], [1, 0]]])

        test.assert_array_equal(sims.numpy(), exp)

    
    def test_pad(self):
        a = np.array([[2, 5, 6, 8, np.nan]])
        b = pad(a, 7, np.nan, axis=1)
        exp = np.array([[2, 5, 6, 8, np.nan, np.nan, np.nan]])
        test.assert_array_equal(b, exp)
        
    def test_stack(self):
        a = np.ones((3, 4, 5))
        b = np.random.randint(2, 9, size=a.shape)
        s = stack(a, b)
        exp = np.zeros((3, 4, 5, 2))
        exp[..., 0] = a
        exp[..., 1] = b
        test.assert_array_equal(s, exp)
    
    def test_unstack(self):
        a = np.ones((3, 4, 5))
        b = np.random.randint(2, 9, size=a.shape)
        s = stack(a, b)
        res_a, res_b = unstack(s)
        test.assert_array_equal(a, res_a)
        test.assert_array_equal(b, res_b)
    
    def test_cutoff_windows(self):        
        data = np.array([[[1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           3,      5,      7, np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                         [[1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                         [[1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           7, np.nan, np.nan, np.nan]]])
        
        acks = np.random.randint(0, 9, size=data.shape)
        a = stack(data, acks)
        self.assertEqual(a.shape, (3, 4, 5, 2))
        data_channel = 0
        result = cutoff_windows(a, data_channel=data_channel)
        
        exp = np.array([[[1, 3, 5, 7, 9],
                         [1, 3, 5, 7, 9]],
                      
                        [[1, 3, 5, 7, 9],
                         [1, 3, 5, 7, 9]],
                      
                        [[1, 3, 5, 7, 9],
                         [1, 3, 5, 7, 9]]])
        test.assert_array_equal(result[..., data_channel], exp)
        
        # remove windows with all acks
        data = np.array([[[1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           3,      5,      7, np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                         [[1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                         [[1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           3,      5,      7,      9],
                          [1,           7, np.nan, np.nan, np.nan]]])
        
        acks = np.random.randint(0, 9, size=data.shape)
        a = stack(data, acks)
        self.assertEqual(a.shape, (3, 4, 5, 2))
        data_channel = 0
        result = cutoff_windows(a, index=0, data_channel=data_channel)
        
        data = np.array([[[1, 3, 5, 7,      9],
                          [1, 3, 5, 7,      9],
                          [1, 3, 5, 7, np.nan]],
                      
                         [[1, 3, 5, 7,      9],
                          [1, 3, 5, 7,      9],
                          [1, 3, 5, 7,      9]],
                      
                         [[1, 3, 5, 7,      9],
                          [1, 3, 5, 7,      9],
                          [1, 3, 5, 7,      9]]])
        test.assert_array_equal(result[..., data_channel], exp)        
    
    def test_discard_along_axes(self):
        data1 = np.array([[[1,           3,      5,      7,      9],
                           [1,           3,      5,      7,      9],
                           [1,           3,      5,      7, np.nan],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                          [[1,           3,      5,      7,      9],
                           [1,           3,      5,      7,      9],
                           [1,           3,      5,      7,      9],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                          [[1,           3,      5,      7,      9],
                           [1,           3,      5,      7,      9],
                           [1,           3,      5,      7,      9],
                           [1,           7, np.nan, np.nan, np.nan]]])
        acks1 = np.random.randint(0, 9, size=data1.shape)
        a1 = stack(data1, acks1)
        self.assertEqual(a1.shape, (3, 4, 5, 2))
        
        data2 = np.array([[[2,           4,      8,      5,      2],
                           [1,           3,      1,      6,      9],
                           [1,           8,      5,      7,      8],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                          [[1,           3,      5,      3,      9],
                           [3,           9,      9,      2,      8],
                           [1,           3,      5,      1,      9],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],
                      
                          [[0,           3,      5,      7,      9],
                           [5,           4,      8,      7,      9],
                           [6,           3,      8,      7,      3],
                           [7,           7, np.nan, np.nan, np.nan]]])
        acks2 = np.random.randint(0, 9, size=data2.shape)
        a2 = stack(data2, acks2)
        self.assertEqual(a2.shape, (3, 4, 5, 2))
        
        res_a1, res_a2 = discard_along_axes(a1, a2)
        exp1 = np.array([[[1, 3, 5, 7, 9],
                          [1, 3, 5, 7, 9]],
                      
                         [[1, 3, 5, 7, 9],
                          [1, 3, 5, 7, 9]],
                      
                         [[1, 3, 5, 7, 9],
                          [1, 3, 5, 7, 9]]])
        test.assert_array_equal(res_a1[..., 0], exp1)
        exp2 = np.array([[[2, 4, 8, 5, 2],
                          [1, 3, 1, 6, 9]],
                      
                         [[1, 3, 5, 3, 9],
                          [3, 9, 9, 2, 8]],
                      
                         [[0, 3, 5, 7, 9],
                          [5, 4, 8, 7, 9]]])
        test.assert_array_equal(res_a2[..., 0], exp2)        
        
        
if __name__ == '__main__':
    unittest.main()
