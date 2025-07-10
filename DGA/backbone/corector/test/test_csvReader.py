import unittest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backbone.corector.CorectionUtilities as cu

'''
readCSV_pitch_or_yaw(fpath:str, inputDim:int, field_v:str|None='Original Pitch', 
        field_gt:str='Ground Truth Pitch',offset:int|None=None) -> Tuple[torch.Tensor, torch.Tensor]

def readCSV_pitch_and_yaw(fpath:str, inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None) \
                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

def readCSV_pitch_and_yaw_together(fpath:str, inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None) \
                    -> Tuple[torch.Tensor, torch.Tensor]:
'''

class test_read_pitch_and_yaw(unittest.TestCase):

    __file=os.path.join(os.path.dirname(os.path.abspath(__file__)),'position_test.csv')

    def test_readCSV_pitch_and_yaw_testPos_defsaultOffset(self):
        p,gp,y,gy=cu.readCSV_pitch_and_yaw(self.__file,5)

        #verify the pitch array
        self.assertEqual(16,len(p),'Didn\'t generated the expected number of groups (pitch).')
        self.assertEqual(5,len(p[0]),'Group dimension other than the reqired one for the first group (pitch).')
        self.assertEqual(5,len(p[1]),'Group dimension other than the reqired one for the second group (pitch).')
        self.assertEqual(5,len(p[2]),'Group dimension other than the reqired one for the third group (pitch).')
        self.assertEqual(5,len(p[15]),'Group dimension other than the reqired one for the last group (pitch).')

        self.assertSequenceEqual([201,202,203,204,205],list(p[0]),'Array not the same as expected (1 pitch).')
        self.assertSequenceEqual([202,203,204,205,206],list(p[1]),'Array not the same as expected (2 pitch).')
        self.assertSequenceEqual([203,204,205,206,207],list(p[2]),'Array not the same as expected (3 pitch).')
        self.assertSequenceEqual([216,217,218,219,220],list(p[15]),'Array not the same as expected (last pitch).')

        #verify the yaw array
        self.assertEqual(16,len(y),'Didn\'t generated the expected number of groups (yaw).')
        self.assertEqual(5,len(y[0]),'Object dimension other than the reqired one for the first group (yaw).')
        self.assertEqual(5,len(y[1]),'Object dimension other than the reqired one for the second group (yaw).')
        self.assertEqual(5,len(y[2]),'Object dimension other than the reqired one for the third group (yaw).')
        self.assertEqual(5,len(y[15]),'Group dimension other than the reqired one for the last group (yaw).')

        self.assertSequenceEqual([501,502,503,504,505],list(y[0]),'Array not the same as expected (1 yaw).')
        self.assertSequenceEqual([502,503,504,505,506],list(y[1]),'Array not the same as expected (2 yaw).')
        self.assertSequenceEqual([503,504,505,506,507],list(y[2]),'Array not the same as expected (3 yaw).')
        self.assertSequenceEqual([516,517,518,519,520],list(y[15]),'Array not the same as expected (last yaw).')

        #verify the pitch gt array
        self.assertSequenceEqual([305,306,307,308,309,310,311,312,313,314,315,316,317, 318, 319, 320],list(gp))

        #verify the yaw gt array
        self.assertSequenceEqual([605,606,607,608,609,610,611,612,613,614,615,616,617, 618, 619, 620],list(gy))

    def test_readCSV_pitch_and_yaw_testPos_Offset4(self):
        p,gp,y,gy=cu.readCSV_pitch_and_yaw(self.__file,5,offset=3)

        #verify the pitch array
        self.assertEqual(16,len(p),'Didn\'t generated the expected number of groups (pitch).')
        self.assertEqual(5,len(p[0]),'Group dimension other than the reqired one for the first group (pitch).')
        self.assertEqual(5,len(p[1]),'Group dimension other than the reqired one for the second group (pitch).')
        self.assertEqual(5,len(p[2]),'Group dimension other than the reqired one for the third group (pitch).')
        self.assertEqual(5,len(p[15]),'Group dimension other than the reqired one for the last group (pitch).')

        self.assertSequenceEqual([201,202,203,204,205],list(p[0]),'Array not the same as expected (1 pitch).')
        self.assertSequenceEqual([202,203,204,205,206],list(p[1]),'Array not the same as expected (2 pitch).')
        self.assertSequenceEqual([203,204,205,206,207],list(p[2]),'Array not the same as expected (3 pitch).')
        self.assertSequenceEqual([216,217,218,219,220],list(p[15]),'Array not the same as expected (last pitch).')

        #verify the yaw array
        self.assertEqual(16,len(y),'Didn\'t generated the expected number of groups (yaw).')
        self.assertEqual(5,len(y[0]),'Object dimension other than the reqired one for the first group (yaw).')
        self.assertEqual(5,len(y[1]),'Object dimension other than the reqired one for the second group (yaw).')
        self.assertEqual(5,len(y[2]),'Object dimension other than the reqired one for the third group (yaw).')
        self.assertEqual(5,len(y[15]),'Group dimension other than the reqired one for the last group (yaw).')

        self.assertSequenceEqual([501,502,503,504,505],list(y[0]),'Array not the same as expected (1 yaw).')
        self.assertSequenceEqual([502,503,504,505,506],list(y[1]),'Array not the same as expected (2 yaw).')
        self.assertSequenceEqual([503,504,505,506,507],list(y[2]),'Array not the same as expected (3 yaw).')
        self.assertSequenceEqual([516,517,518,519,520],list(y[15]),'Array not the same as expected (last yaw).')

        #verify the pitch gt array
        self.assertSequenceEqual([304, 305,306,307,308,309,310,311,312,313,314,315,316,317, 318, 319],list(gp))

        #verify the yaw gt array
        self.assertSequenceEqual([604, 605,606,607,608,609,610,611,612,613,614,615,616,617, 618, 619],list(gy))

    def test_readCSV_pitch_and_yaw_testPos_Offsetfirst(self):
        p,gp,y,gy=cu.readCSV_pitch_and_yaw(self.__file,inputDim=3,offset=0)

        #verify the pitch array
        self.assertEqual(18,len(p),'Didn\'t generated the expected number of groups (pitch).')
        self.assertEqual(3,len(p[0]),'Group dimension other than the reqired one for the first group (pitch).')
        self.assertEqual(3,len(p[1]),'Group dimension other than the reqired one for the second group (pitch).')
        self.assertEqual(3,len(p[2]),'Group dimension other than the reqired one for the third group (pitch).')
        self.assertEqual(3,len(p[15]),'Group dimension other than the reqired one for the last group (pitch).')

        self.assertSequenceEqual([201,202,203],list(p[0]),'Array not the same as expected (1 pitch).')
        self.assertSequenceEqual([202,203,204],list(p[1]),'Array not the same as expected (2 pitch).')
        self.assertSequenceEqual([203,204,205],list(p[2]),'Array not the same as expected (3 pitch).')
        self.assertSequenceEqual([218,219,220],list(p[17]),'Array not the same as expected (last pitch).')

        #verify the yaw array
        self.assertEqual(18,len(y),'Didn\'t generated the expected number of groups (yaw).')
        self.assertEqual(3,len(y[0]),'Object dimension other than the reqired one for the first group (yaw).')
        self.assertEqual(3,len(y[1]),'Object dimension other than the reqired one for the second group (yaw).')
        self.assertEqual(3,len(y[2]),'Object dimension other than the reqired one for the third group (yaw).')
        self.assertEqual(3,len(y[15]),'Group dimension other than the reqired one for the last group (yaw).')

        self.assertSequenceEqual([501,502,503],list(y[0]),'Array not the same as expected (1 yaw).')
        self.assertSequenceEqual([502,503,504],list(y[1]),'Array not the same as expected (2 yaw).')
        self.assertSequenceEqual([503,504,505],list(y[2]),'Array not the same as expected (3 yaw).')
        self.assertSequenceEqual([518,519,520],list(y[17]),'Array not the same as expected (last yaw).')

        #verify the pitch gt array
        self.assertSequenceEqual([301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318],list(gp))

        #verify the yaw gt array
        self.assertSequenceEqual([601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618],list(gy))

    def test_readCSV_pitch_and_yaw_test_gr1val(self):
        p,gp,y,gy=cu.readCSV_pitch_and_yaw(self.__file,inputDim=1)

        #verify the pitch array
        self.assertEqual(20,len(p),'Didn\'t generated the expected number of groups (pitch).')
        self.assertEqual(1,len(p[0]),'Group dimension other than the reqired one for the first group (pitch).')
        self.assertEqual(1,len(p[1]),'Group dimension other than the reqired one for the second group (pitch).')
        self.assertEqual(1,len(p[2]),'Group dimension other than the reqired one for the third group (pitch).')
        self.assertEqual(1,len(p[19]),'Group dimension other than the reqired one for the last group (pitch).')

        self.assertSequenceEqual([201],list(p[0]),'Array not the same as expected (1 pitch).')
        self.assertSequenceEqual([202],list(p[1]),'Array not the same as expected (2 pitch).')
        self.assertSequenceEqual([203],list(p[2]),'Array not the same as expected (3 pitch).')
        self.assertSequenceEqual([220],list(p[19]),'Array not the same as expected (last pitch).')

        #verify the yaw array
        self.assertEqual(20,len(y),'Didn\'t generated the expected number of groups (yaw).')
        self.assertEqual(1,len(y[0]),'Object dimension other than the reqired one for the first group (yaw).')
        self.assertEqual(1,len(y[1]),'Object dimension other than the reqired one for the second group (yaw).')
        self.assertEqual(1,len(y[2]),'Object dimension other than the reqired one for the third group (yaw).')
        self.assertEqual(1,len(y[19]),'Group dimension other than the reqired one for the last group (yaw).')

        self.assertSequenceEqual([501],list(y[0]),'Array not the same as expected (1 yaw).')
        self.assertSequenceEqual([502],list(y[1]),'Array not the same as expected (2 yaw).')
        self.assertSequenceEqual([503],list(y[2]),'Array not the same as expected (3 yaw).')
        self.assertSequenceEqual([520],list(y[19]),'Array not the same as expected (last yaw).')

        #verify the pitch gt array
        self.assertSequenceEqual([301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320],list(gp))

        #verify the yaw gt array
        self.assertSequenceEqual([601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620],list(gy))


class test_pitch_and_yaw_together(unittest.TestCase):
    __file=os.path.join(os.path.dirname(os.path.abspath(__file__)),'position_test.csv')

    def test_readCSV_pitch_and_yaw_testPos5_defaultOffset(self):
        p,gp=cu.readCSV_pitch_and_yaw_together(self.__file,5)

        #verify the pitch array
        self.assertEqual(16,len(p),'Didn\'t generated the expected number of groups (pitch,yaw).')
        self.assertEqual(10,len(p[0]),'Group dimension other than the reqired one for the first group (pitch,yaw).')
        self.assertEqual(10,len(p[1]),'Group dimension other than the reqired one for the second group (pitch,yaw).')
        self.assertEqual(10,len(p[2]),'Group dimension other than the reqired one for the third group (pitch,yaw).')
        self.assertEqual(10,len(p[15]),'Group dimension other than the reqired one for the last group (pitch,yaw).')

        self.assertSequenceEqual([201,202,203,204,205,501,502,503,504,505],list(p[0]),'Array not the same as expected (1 pitch,yaw).')
        self.assertSequenceEqual([202,203,204,205,206,502,503,504,505,506],list(p[1]),'Array not the same as expected (2 pitch,yaw).')
        self.assertSequenceEqual([203,204,205,206,207,503,504,505,506,507],list(p[2]),'Array not the same as expected (3 pitch,yaw).')
        self.assertSequenceEqual([216,217,218,219,220,516,517,518,519,520],list(p[15]),'Array not the same as expected (last pitch,yaw).')

        self.assertEqual(16,len(gp),'Didn\'t generated the expected number of groups (pitch,yaw).')
        self.assertSequenceEqual([305,605],list(gp[0]))
        self.assertSequenceEqual([306,606],list(gp[1]))
        self.assertSequenceEqual([307,607],list(gp[2]))
        self.assertSequenceEqual([320,620],list(gp[15]))

    def test_readCSV_pitch_and_yaw_testPos5_Offset1(self):
        p,gp=cu.readCSV_pitch_and_yaw_together(self.__file,5,offset=2)

        #verify the pitch array
        self.assertEqual(16,len(p),'Didn\'t generated the expected number of groups (pitch,yaw).')
        self.assertEqual(10,len(p[0]),'Group dimension other than the reqired one for the first group (pitch,yaw).')
        self.assertEqual(10,len(p[1]),'Group dimension other than the reqired one for the second group (pitch,yaw).')
        self.assertEqual(10,len(p[2]),'Group dimension other than the reqired one for the third group (pitch,yaw).')
        self.assertEqual(10,len(p[15]),'Group dimension other than the reqired one for the last group (pitch,yaw).')

        self.assertSequenceEqual([201,202,203,204,205,501,502,503,504,505],list(p[0]),'Array not the same as expected (1 pitch,yaw).')
        self.assertSequenceEqual([202,203,204,205,206,502,503,504,505,506],list(p[1]),'Array not the same as expected (2 pitch,yaw).')
        self.assertSequenceEqual([203,204,205,206,207,503,504,505,506,507],list(p[2]),'Array not the same as expected (3 pitch,yaw).')
        self.assertSequenceEqual([216,217,218,219,220,516,517,518,519,520],list(p[15]),'Array not the same as expected (last pitch,yaw).')

        self.assertEqual(16,len(gp),'Didn\'t generated the expected number of groups (pitch,yaw).')
        self.assertSequenceEqual([303,603],list(gp[0]))
        self.assertSequenceEqual([304,604],list(gp[1]))
        self.assertSequenceEqual([305,605],list(gp[2]))
        self.assertSequenceEqual([318,618],list(gp[15]))

    def test_readCSV_pitch_and_yaw_testPos7_Offset1(self):
        p,gp=cu.readCSV_pitch_and_yaw_together(self.__file,7,offset=2)

        #verify the pitch array
        self.assertEqual(14,len(p),'Didn\'t generated the expected number of groups (pitch,yaw).')
        self.assertEqual(14,len(p[0]),'Group dimension other than the reqired one for the first group (pitch,yaw).')
        self.assertEqual(14,len(p[1]),'Group dimension other than the reqired one for the second group (pitch,yaw).')
        self.assertEqual(14,len(p[2]),'Group dimension other than the reqired one for the third group (pitch,yaw).')
        self.assertEqual(14,len(p[13]),'Group dimension other than the reqired one for the last group (pitch,yaw).')

        self.assertSequenceEqual([201,202,203,204,205,206,207,501,502,503,504,505,506,507],list(p[0]),'Array not the same as expected (1 pitch,yaw).')
        self.assertSequenceEqual([202,203,204,205,206,207,208,502,503,504,505,506,507,508],list(p[1]),'Array not the same as expected (2 pitch,yaw).')
        self.assertSequenceEqual([203,204,205,206,207,208,209,503,504,505,506,507,508,509],list(p[2]),'Array not the same as expected (3 pitch,yaw).')
        self.assertSequenceEqual([214,215,216,217,218,219,220,514,515,516,517,518,519,520],list(p[13]),'Array not the same as expected (last pitch,yaw).')

        self.assertEqual(14,len(gp),'Didn\'t generated the expected number of groups (pitch,yaw).')
        self.assertSequenceEqual([303,603],list(gp[0]))
        self.assertSequenceEqual([304,604],list(gp[1]))
        self.assertSequenceEqual([305,605],list(gp[2]))
        self.assertSequenceEqual([316,616],list(gp[13]))
        


if __name__ == '__main__':
    unittest.main()
