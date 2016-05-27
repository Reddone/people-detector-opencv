import cv2
import numpy as np
import random
import pickle
import os
import re
import json
import xml.etree.ElementTree as Et

"""------------------------------------------------------------------------------------------------------------------"""


def load_hog(hog_file):
    """Loads the HoG module"""
    if os.path.isfile(hog_file):
        hog_obj = cv2.HOGDescriptor(hog_file)
    else:
        hog_obj = cv2.HOGDescriptor()
        hog_obj.save('hog.xml')
    return hog_obj


def load_svm(svm_file, x, y, svm_par):
    """Loads the SVM module"""
    if os.path.isfile(svm_file):
        svm_vec = pickle.load(open(svm_file))
    else:
        svm_obj = cv2.SVM()
        svm_obj.train_auto(x, y, None, None, params=svm_par, k_fold=5, balanced=True)
        # svm_obj.train(x, y, None, None, params=svm_par)
        svm_obj.save("svm.xml")
        tree = Et.parse('svm.xml')
        root = tree.getroot()
        sup_vec = root.getchildren()[0].getchildren()[-2].getchildren()[0]
        rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
        svm_vec = [float(x) for x in re.sub('\s+', ' ', sup_vec.text).strip().split(' ')]
        svm_vec.append(-rho)
        pickle.dump(svm_vec, open(svm_file, 'w'))
    return svm_vec


"""------------------------------------------------------------------------------------------------------------------"""


def get_pos(pos_path, size=(128, 64)):
    """Creates a list which holds all the positive examples"""
    pos_list = []
    pos_lab = []
    for name in os.listdir(pos_path):
        path = os.path.join(pos_path, name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        crop = cv2.resize(image, (size[1], size[0]))
        # crop = cv2.GaussianBlur(crop, (0, 0), 1)
        pos_list.append(crop)
        pos_lab.append('0')
    return pos_list, pos_lab


def get_neg_fix(neg_path, size=(128, 64)):
    """Creates a list which holds all the negative examples"""
    neg_list = []
    neg_lab = []
    for name in os.listdir(neg_path):
        path = os.path.join(neg_path, name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        crop = cv2.resize(image, (size[1], size[0]))
        # crop = cv2.GaussianBlur(crop, (0, 0), 1)
        neg_list.append(crop)
        neg_lab.append('1')
    return neg_list, neg_lab


def get_neg_rnd(neg_path, size=(128, 64), scale=0.65):
    """Creates a list which holds all the negative examples"""
    neg_list = []
    neg_lab = []
    for name in os.listdir(neg_path):
        path = os.path.join(neg_path, name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        while image.shape[0] > size[0] and image.shape[1] > size[1]:
            x = random.randint(0, image.shape[0]-size[0])
            y = random.randint(0, image.shape[1]-size[1])
            crop = image[x:x+size[0], y:y+size[1]]
            # crop = cv2.GaussianBlur(crop, (0, 0), 1)
            neg_list.append(crop)
            neg_lab.append('1')
            image = cv2.resize(image, (0, 0), None, scale, scale)
    return neg_list, neg_lab


def get_frame(frm_path):
    """Creates a list which holds the frames of the test video"""
    frm_list = []
    names = sorted(os.listdir(frm_path))
    for name in names:
        path = os.path.join(frm_path, name)
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        frm_list.append(frame)
    return frm_list


def get_hog(pos_list, pos_lab, neg_list, neg_lab, hog_obj):
    """Creates a list which holds all the hogs with their respective labels"""
    hog_list = []
    for i in range(len(pos_list)):
        hog_list.append((hog_obj.compute(pos_list[i]), pos_lab[i]))
    for i in range(len(neg_list)):
        hog_list.append((hog_obj.compute(neg_list[i]), neg_lab[i]))
    random.shuffle(hog_list)
    return hog_list


def get_data(hog_list):
    """Creates the training set for the SVM"""
    m = len(hog_list)
    n = 3780
    x = np.zeros((m, n), np.float32)
    y = np.zeros(m, np.float32)
    for i in range(m):
        hog = hog_list[i]
        x[i, :] = np.transpose(hog[0])
        y[i] = hog[1]
    return x, y


"""------------------------------------------------------------------------------------------------------------------"""


def is_inside(r, q):
    """Checks if rectangle r is inside rectangle q"""
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rectangles, color, thickness=1):
    """Draws the rectangles"""
    for x, y, w, h in rectangles:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def rescale(rectangles):
    """Rescale rectangles"""
    new_rectangles = []
    for x, y, w, h in rectangles:
        pad_x, pad_y = int(0.15*w), int(0.05*h)
        new_rectangles.append([x+pad_x, y+pad_y, int(0.85*w)-pad_x, int(0.95*h)-pad_y])
    return new_rectangles


def intersection(r, q):
    """Intersection between rectangles r and q"""
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    left = max(rx, qx)
    right = min(rx+rw, qx+qw)
    top = max(ry, qy)
    bottom = min(ry+rh, qy+qh)
    intersection_area = (right-left)*(bottom-top)
    return intersection_area*(int(intersection_area > 0))


def union(r, q):
    """Union between rectangles r and q"""
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    ar = rw*rh
    aq = qw*qh
    union_area = ar+aq-intersection(r, q)
    return union_area


def get_truth(truth_file):
    """"Creates the ground truth from annotations"""
    true_detection_list = []
    with open(truth_file) as f:
        annotations = json.load(f)
        for i in range(len(annotations)):
            found = []
            for j in range(len(annotations[i]['annotations'])):
                x = annotations[i]['annotations'][j]['x']
                y = annotations[i]['annotations'][j]['y']
                w = annotations[i]['annotations'][j]['width']
                h = annotations[i]['annotations'][j]['height']
                r = [int(x), int(y), int(w), int(h)]
                found.append(np.asarray(r, np.int32))
            true_detection_list.append(found)
    return true_detection_list


def get_performances(hog_detection_list, true_detection_list, n_frame,  threshold=0.4):
    """Computes the performances for the people detector"""
    prc = 0.0
    rec = 0.0
    for i in range(n_frame):
        tp = 0.0
        fp = 0.0
        r_list = hog_detection_list[i]
        q_list = true_detection_list[i]
        for r in r_list:
            fp_flag = True
            for q in q_list:
                int_over_uni = float(intersection(r, q))/float(union(r, q))
                if int_over_uni > threshold:
                    tp += 1.0
                    fp_flag = False
                    break
            if fp_flag:
                fp += 1
        fn = len(q_list) - tp
        try:
            prc += tp/(tp+fp)
        except ZeroDivisionError:
            prc += 0
        try:
            rec += tp/(tp+fn)
        except ZeroDivisionError:
            rec += 0
    prc /= n_frame
    rec /= n_frame
    fsc = (2*prc*rec)/(prc+rec)
    return prc, rec, fsc
