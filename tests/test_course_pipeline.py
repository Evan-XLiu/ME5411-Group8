import numpy as np
from task_03.crop import crop_roi
from task_04.binarize import otsu_binarization
from task_06.segment import segment_character_boxes, normalize_character_crop

def test_binarize_separates_two_levels():
    image=np.array([[0,0,255,255]],dtype=np.uint8)
    _,mask=otsu_binarization(image)
    assert mask.tolist()==[[0,0,255,255]]

def test_segment_orders_components_and_normalizes():
    mask=np.zeros((20,40),dtype=np.uint8);mask[3:17,2:9]=255;mask[3:17,22:29]=255
    boxes,_,_=segment_character_boxes(mask,expected_count=2)
    assert boxes==[(2,3,7,14),(22,3,7,14)]
    assert normalize_character_crop(mask).shape==(96,96)
