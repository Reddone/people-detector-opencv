import cv2
import numpy as np
import utils


"""Enable drawing and performances"""
do_draw = True
do_perf = False

"""Training images"""
pos_path = r'/home/skynet/Datasets/PeopleDataset/Positive'
neg_path = r'/home/skynet/Datasets/PeopleDataset/Negative'

"""Test videos"""
frm_path = r'/home/skynet/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_008'
# frm_path = r'/home/skynet/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_007'
# frm_path = r'/home/skynet/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_006'
# frm_path = r'/home/skynet/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_005'
# frm_path = r'/home/skynet/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_001'

"""Get positive and negative examples"""
pos_list, pos_lab = utils.get_pos(pos_path)
# neg_list, neg_lab = utils.get_neg_fix(neg_path)
neg_list, neg_lab = utils.get_neg_rnd(neg_path)
del pos_path, neg_path

"""Initialize HOG"""
hog_obj = utils.load_hog('hog.xml')
hog_list = utils.get_hog(pos_list, pos_lab, neg_list, neg_lab, hog_obj)
del pos_list, pos_lab, neg_list, neg_lab
# hog_par = {'winStride': (8, 8), 'padding': (0, 0), 'scale': 1.2}
# hog_par = {'hitThreshold': 1.2, 'winStride': (8, 8), 'padding': (0, 0), 'scale': 1.2, 'finalThreshold': 4}
hog_par = {'hitThreshold': 1.4, 'winStride': (8, 8), 'padding': (0, 0), 'scale': 1.2, 'finalThreshold': 2}

"""Initialize and train the SVM"""
x, y = utils.get_data(hog_list)
del hog_list
svm_par = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)
# svm_par = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=0.01)
svm_vec = utils.load_svm("svm.pickle", x, y, svm_par)
# svm_vec = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog_obj.setSVMDetector(np.array(svm_vec))

"""Ground truth"""
true_detection_list = utils.get_truth('view8.json')
n_frame = len(true_detection_list)

"""Multi-scale detector on the video"""
hog_detection_list = []
frm_list = utils.get_frame(frm_path)
frm_list = frm_list[0:n_frame]
i = -1
for frm in frm_list:
    i += 1
    found_true = true_detection_list[i]
    found_filtered = []
    found, w = hog_obj.detectMultiScale(frm, **hog_par)
    for r in found:
        inside = False
        for q in found:
            if utils.is_inside(r, q):
                inside = True
                break
        if not inside:
            found_filtered.append(r)
    found_hog = utils.rescale(found_filtered)
    hog_detection_list.append(found_hog)
    if do_draw:
        frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
        # utils.draw_detections(frm, found, (0, 0, 255), 1)
        # utils.draw_detections(frm, found_filtered, (0, 0, 255), 3)
        utils.draw_detections(frm, found_hog, (0, 0, 255), 3)
        utils.draw_detections(frm, found_true, (0, 255, 0), 1)
        cv2.imshow('People Detector', frm)
        cv2.waitKey()

"""Performances indices"""
if do_perf:
    precision, recall, f_score = utils.get_performances(hog_detection_list, true_detection_list, n_frame)
    print precision, recall, f_score
