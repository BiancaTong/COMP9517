from __future__ import print_function
import cv2
import argparse
import os
import numpy as np
import random


def up_to_step_1(imgs):
    """Complete pipeline up to step 1: Detecting features and descriptors"""

    # construct a DoG keypoint detector and a SURF feature extractor
    surf_detector = cv2.FeatureDetector_create("SURF")
    surf_detector.setInt("hessianThreshold",1500)
    surf_extractor=cv2.DescriptorExtractor_create("SURF")

    for img in imgs:
        # convert to gray image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect keypoints in the image
        kp = surf_detector.detect(img_gray, None)
        # extract features from the image
        (kp, features) = surf_extractor.compute(img_gray, kp)
        # draw features on image
        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return imgs


def save_step_1(imgs, img_names, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    output_path='./output/step1'
    for i in range(0, len(imgs)):
        cv2.imwrite(os.path.join(output_path, img_names[i]), imgs[i])


def up_to_step_2(imgs):

    """Complete pipeline up to step 1: Detecting features and descriptors"""

    # construct a DoG keypoint detector and a SURF feature extractor
    surf_detector = cv2.FeatureDetector_create("SURF")
    surf_detector.setInt("hessianThreshold",1500)
    surf_extractor=cv2.DescriptorExtractor_create("SURF")

    kp_list = []
    feature_list = []
    for img in imgs:
        # convert to gray image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect keypoints in the image
        kp = surf_detector.detect(img_gray)
        # extract features from the image
        (kp, features) = surf_extractor.compute(img_gray, kp)
        # draw features on image
        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # convert keypoints objects to numpy
        kp = np.float32([k.pt for k in kp])
        kp_list.append(kp)
        feature_list.append(features)

    """Complete pipeline up to step 2: Calculate matching feature points"""
    match_list = []
    modified_imgs = []
    exist_list = []
    for i in range(0,len(imgs)):
        for j in range(0, len(imgs)):
            if [i,j] not in exist_list and [j,i] not in exist_list:
                exist_list.append([i,j])
            else:
                continue

            # not match with itself
            if i != j:
                m_list = [i]
                m_list.append(j)
                m_list.append(len(kp_list[i]))
                m_list.append(len(kp_list[j]))

                # match two images using features KNN
                D_list = []
                K_list = []
                for f in range(0, len(feature_list[i])):
                    f_i = feature_list[i][f]
                    d_list = []
                    k_list = []
                    for ff in range(0, len(feature_list[j])):
                        f_j = feature_list[j][ff]
                        d = np.linalg.norm(f_i - f_j)
                        if len(d_list) < 2:
                            d_list.append(d)
                            k_list.append([kp_list[i][f], kp_list[j][ff]])
                        else:
                            if d < max(d_list):
                                if d_list[0] == max(d_list):
                                    d_list.remove(d_list[0])
                                    k_list = k_list[1:]
                                else:
                                    d_list.remove(d_list[1])
                                    k_list = k_list[0:1]
                                d_list.append(d)
                                k_list.append([kp_list[i][f], kp_list[j][ff]])
                    D_list.append(d_list)
                    K_list.append(k_list)

                # ensure the distance is within a certain ratio of each--Lowe's ratio test
                good_match = []
                for x in range(0, len(D_list)):
                    if D_list[x][0] < D_list[x][1]:
                        if D_list[x][0] < 0.75 * D_list[x][1]:
                            good_match.append(K_list[x][0])
                    else:
                        if D_list[x][1] < 0.75 * D_list[x][0]:
                            good_match.append(K_list[x][1])
                if len(good_match) != 0:
                    m_list.append(len(good_match))
                    match_list.append(m_list)

                    # combine two images
                    h1, w1, c1 = imgs[i].shape
                    h2, w2, c2 = imgs[j].shape
                    img_xy = np.zeros((max(h1, h2),w1+w2,3), np.uint8)
                    for q in range(0, max(h1,h2)):
                	    img_xy[q,:w1] = imgs[i][q]
                	    img_xy[q,w1:] = imgs[j][q]

                    # draw match line
                    for q in range(0, len(good_match)):
                        p1 = good_match[q][0]
                        p2 = good_match[q][1]
                        p1_x = p1[0]
                        p1_y = p1[1]
                        p2_x = w1 + p2[0]
                        p2_y = p2[1]
                        cv2.line(img_xy, (int(p1_x),int(p1_y)), (int(p2_x),int(p2_y)), (0,255,255))
                    modified_imgs.append(img_xy)
    return modified_imgs, match_list
    #return imgs, [[0,1,70,70,40],[0,2,70,70,40],[1,2,70,70,40]]


def save_step_2(imgs, match_list, img_names, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    output_path="./output/step2"
    n = 0
    for match_pair in match_list:
        x = match_pair[0] #index of img1
        y = match_pair[1] #index of img2
        num_x = match_pair[2]
        num_y = match_pair[3]
        num_xy = match_pair[4]
        xy_name = img_names[x][:-4] + '_' + str(num_x) + '_' + img_names[y][:-4]+ '_' + str(num_y) + '_' + str(num_xy) + '.jpg'
        cv2.imwrite(os.path.join(output_path, xy_name), imgs[n])
        n += 1



def up_to_step_3(imgs):
    """Complete pipeline up to step 1: Detecting features and descriptors"""

    # construct a DoG keypoint detector and a SURF feature extractor
    surf_detector = cv2.FeatureDetector_create("SURF")
    surf_detector.setInt("hessianThreshold",1500)
    surf_extractor=cv2.DescriptorExtractor_create("SURF")

    kp_list = []
    feature_list = []
    for img in imgs:
        # convert to gray image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect keypoints in the image
        kp = surf_detector.detect(img_gray)
        # extract features from the image
        (kp, features) = surf_extractor.compute(img_gray, kp)
        # convert keypoints objects to numpy
        kp = np.float32([k.pt for k in kp])
        kp_list.append(kp)
        feature_list.append(features)

    """Complete pipeline up to step 2: Calculate matching feature points"""

    exist_list = []
    img_pairs = []
    for i in range(0,len(imgs)):
        for j in range(0, len(imgs)):
            if [i,j] not in exist_list and [j,i] not in exist_list:
                exist_list.append([i,j])
            else:
                continue
            # not match with itself
            if i != j:
                # match two images using features KNN
                D_list = []
                K_list = []
                for f in range(0, len(feature_list[i])):
                    f_i = feature_list[i][f]
                    d_list = []
                    k_list = []
                    for ff in range(0, len(feature_list[j])):
                        f_j = feature_list[j][ff]
                        d = np.linalg.norm(f_i - f_j)
                        if len(d_list) < 2:
                            d_list.append(d)
                            k_list.append([kp_list[i][f], kp_list[j][ff]])
                        else:
                            if d < max(d_list):
                                if d_list[0] == max(d_list):
                                    d_list.remove(d_list[0])
                                    k_list = k_list[1:]
                                else:
                                    d_list.remove(d_list[1])
                                    k_list = k_list[0:1]
                                d_list.append(d)
                                k_list.append([kp_list[i][f], kp_list[j][ff]])
                    D_list.append(d_list)
                    K_list.append(k_list)

                # ensure the distance is within a certain ratio of each--Lowe's ratio test
                good_match = []
                for x in range(0, len(D_list)):
                    if D_list[x][0] < D_list[x][1]:
                        if D_list[x][0] < 0.75 * D_list[x][1]:
                            good_match.append(K_list[x][0])
                    else:
                        if D_list[x][1] < 0.75 * D_list[x][0]:
                            good_match.append(K_list[x][1])

                """Complete pipeline up to step 3: estimating homographies and warpings"""
                # RANSAC
                best_h = 0
                max_num = []
                for n in range(1000):
                    # ransom find 4 keypoints to calculate homography
                    p1 = good_match[random.randrange(0, len(good_match))]
                    p2 = good_match[random.randrange(0, len(good_match))]
                    p3 = good_match[random.randrange(0, len(good_match))]
                    p4 = good_match[random.randrange(0, len(good_match))]
                    #print(p)
                    p = [p1, p2, p3, p4]
                    # caculate homography on four keypoints
                    a = []
                    for pp in p:
                        #p11 = np.matrix([pp.item(0),pp.item(1),1])
                        #p22 = np.matrix([pp.item(2),pp.item(3),1])
                        p11 = [pp[0][0], pp[0][1], 1]
                        p22 = [pp[1][0], pp[1][1], 1]
                        # SVD the last cotumn v is h
                        # H * p11 = p22
                        #a.append([-p22.item(2) * p11.item(0), -p22.item(2) * p11.item(1), -p22.item(2) * p11.item(2), 0, 0, 0, p22.item(0) * p11.item(0), p22.item(0) * p11.item(1), p22.item(0) * p11.item(2)])
                        #a.append([0, 0, 0, -p22.item(2) * p11.item(0), -p22.item(2) * p11.item(1), -p22.item(2) * p11.item(2), p22.item(1) * p11.item(0), p22.item(1) * p11.item(1), p22.item(1) * p11.item(2)])
                        a.append([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]])
                        a.append([0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                        #A = np.matric([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]], [0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                    A = np.matrix(a)
                    u, s, v = np.linalg.svd(A)
                    h = np.reshape(v[8], (3, 3))
                    h = (1/h.item(8))*h
                    # caculate the error between line and points
                    inliers = []
                    for q in range(0, len(good_match)):
                        p1_x = good_match[q][0][0]
                        p1_y = good_match[q][0][1]
                        p2_x = good_match[q][1][0]
                        p2_y = good_match[q][1][1]
                        p11 = np.transpose(np.matrix([p1_x, p1_y, 1]))
                        p11_pred = np.dot(h, p11)
                        p11_pred = (1/p11_pred.item(2))*p11_pred
                        p22 = np.transpose(np.matrix([p2_x, p2_y, 1]))
                        err = p22 - p11_pred
                        if np.linalg.norm(err) < 6:
                            inliers.append(q)
                    # whether found best homography
                    if len(inliers) > len(good_match)*0.60:
                        best_h = h
                        break
                    if len(inliers) > len(max_num):
                        max_num = inliers[:]
                        best_h = h
                #print(best_h)
                img_pair = []
                best_hh = best_h.I
                best_hh = (1/best_hh.item(8))*best_hh
                #best_hh = np.linalg.inv(HH)
                #print(best_hh)
                # backward wraping image2 based on image1 right_shift
                h1, w1, c1 = imgs[i].shape
                h2, w2, c2 = imgs[j].shape
                conor2 = [[0,0],[0,w2-1],[h2-1,0],[h2-1,w2-1]]
                img2_h = []
                img2_w = []
                for p in conor2:
                    p2 = np.transpose(np.matrix([p[1], p[0], 1])) # homogeneous representation
                    p1 = np.dot(best_hh, p2)
                    p1 = (1/p1.item(2))*p1
                    x = int(p1.item(0))
                    y = int(p1.item(1))
                    img2_h.append(y)
                    img2_w.append(x)
                h2_new = max(img2_h)-min(img2_h)+1
                w2_new = max(img2_w)-min(img2_w)+1
                img_2 = np.zeros((h2_new,w2_new,3), np.uint8)
                for y in range(0, h2):
                    for x in range(0, w2):
                        p2 = np.transpose(np.matrix([x, y, 1])) # homogeneous representation
                        p1 = np.dot(best_hh, p2)
                        p1 = (1/p1.item(2))*p1
                        xx = int(p1.item(0)-min(img2_w))
                        yy = int(p1.item(1)-min(img2_h))
                        if xx>w2_new-1 or yy>h2_new-1:
                            break
                        if xx<0 or yy<0:
                            break
                        img_2[yy][xx] = imgs[j][y][x]
                #cv2.imwrite('./output/step3/2.jpg', img_2)
                #cv2.waitKey(0)

                # backward wraping image1 based on image2 left_shift
                conor1 = [[0,0],[0,w1-1],[h1-1,0],[h1-1,w1-1]]
                img1_h = []
                img1_w = []
                for p in conor1:
                    p2 = np.transpose(np.matrix([p[1], p[0], 1])) # homogeneous representation
                    p1 = np.dot(best_h, p2)
                    p1 = (1/p1.item(2))*p1
                    x = int(p1.item(0))
                    y = int(p1.item(1))
                    img1_h.append(y)
                    img1_w.append(x)
                h1_new = max(img1_h)-min(img1_h)+1
                w1_new = max(img1_w)-min(img1_w)+1
                img_1 = np.zeros((h1_new,w1_new,3), np.uint8)
                for y in range(0, h1):
                    for x in range(0, w1):
                        p1 = np.transpose(np.matrix([x, y, 1])) # homogeneous representation
                        p2 = np.dot(best_h, p1)
                        p2 = (1/p2.item(2))*p2
                        xx = int(p2[0])-min(img1_w)
                        yy = int(p2[1])-min(img1_h)
                        if xx>w1_new-1 or yy>h1_new-1:
                            break
                        if xx<0 or yy<0:
                            break
                        img_1[yy][xx] = imgs[i][y][x]
                img_pair.append(img_2)
                img_pair.append([j,i])
                img_pair.append(img_1)
                img_pair.append([i,j])
                img_pairs.append(img_pair)
    return img_pairs


def save_step_3(img_pairs, img_names, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    output_path="./output/step3"
    for img_pair in img_pairs:
        img2 = img_pair[0]
        img2_name = img_names[img_pair[1][0]][:-4] + '_based on_' +  img_names[img_pair[1][1]][:-4] + '.jpg'
        img1 = img_pair[2]
        img1_name = img_names[img_pair[3][0]][:-4] + '_based on_' +  img_names[img_pair[3][1]][:-4] + '.jpg'
        cv2.imwrite(os.path.join(output_path, img2_name), img2)
        cv2.imwrite(os.path.join(output_path, img1_name), img1)




def up_to_step_4(imgs):

    """Complete pipeline up to step 1: Detecting features and descriptors"""
    # construct a DoG keypoint detector and a SURF feature extractor
    surf_detector = cv2.FeatureDetector_create("SURF")
    surf_detector.setInt("hessianThreshold",1500)
    surf_extractor=cv2.DescriptorExtractor_create("SURF")

    kp_list = []
    feature_list = []
    for img in imgs:
        # convert to gray image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect keypoints in the image
        kp = surf_detector.detect(img_gray)
        # extract features from the image
        (kp, features) = surf_extractor.compute(img_gray, kp)
        # convert keypoints objects to numpy
        kp = np.float32([k.pt for k in kp])
        kp_list.append(kp)
        feature_list.append(features)

    """Complete the pipeline and generate a panoramic image"""
    img_left = []
    img_right = []
    img_mid = []
    if len(imgs)%2 == 0:
        # left_shift
        for i in range(0,len(imgs)//2):
            j = i+1
            h1, w1, c1 = imgs[i].shape
            h2, w2, c2 = imgs[j].shape
            # match two images using features KNN
            D_list = []
            K_list = []
            for f in range(0, len(feature_list[i])):
                f_i = feature_list[i][f]
                d_list = []
                k_list = []
                for ff in range(0, len(feature_list[j])):
                    f_j = feature_list[j][ff]
                    d = np.linalg.norm(f_i - f_j)
                    if len(d_list) < 2:
                        d_list.append(d)
                        k_list.append([kp_list[i][f], kp_list[j][ff]])
                    else:
                        if d < max(d_list):
                            if d_list[0] == max(d_list):
                                d_list.remove(d_list[0])
                                k_list = k_list[1:]
                            else:
                                d_list.remove(d_list[1])
                                k_list = k_list[0:1]
                            d_list.append(d)
                            k_list.append([kp_list[i][f], kp_list[j][ff]])
                D_list.append(d_list)
                K_list.append(k_list)

            # ensure the distance is within a certain ratio of each--Lowe's ratio test
            good_match = []
            for x in range(0, len(D_list)):
                if D_list[x][0] < D_list[x][1]:
                    if D_list[x][0] < 0.75 * D_list[x][1]:
                        good_match.append(K_list[x][0])
                else:
                    if D_list[x][1] < 0.75 * D_list[x][0]:
                        good_match.append(K_list[x][1])
            # RANSAC
            best_h = 0
            max_num = []
            for n in range(1000):
                # ransom find 4 keypoints to calculate homography
                p1 = good_match[random.randrange(0, len(good_match))]
                p2 = good_match[random.randrange(0, len(good_match))]
                p3 = good_match[random.randrange(0, len(good_match))]
                p4 = good_match[random.randrange(0, len(good_match))]
                #print(p)
                p = [p1, p2, p3, p4]
                # caculate homography on four keypoints
                a = []
                for pp in p:
                    p11 = [pp[0][0], pp[0][1], 1]
                    p22 = [pp[1][0], pp[1][1], 1]
                    # SVD the last cotumn v is h
                    # H * p11 = p22
                    a.append([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]])
                    a.append([0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                    #A = np.matric([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]], [0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                A = np.matrix(a)
                u, s, v = np.linalg.svd(A)
                h = np.reshape(v[8], (3, 3))
                h = (1/h.item(8))*h
                # caculate the error between line and points
                inliers = []
                for q in range(0, len(good_match)):
                    p1_x = good_match[q][0][0]
                    p1_y = good_match[q][0][1]
                    p2_x = good_match[q][1][0]
                    p2_y = good_match[q][1][1]
                    p11 = np.transpose(np.matrix([p1_x, p1_y, 1]))
                    p11_pred = np.dot(h, p11)
                    p11_pred = (1/p11_pred.item(2))*p11_pred
                    p22 = np.transpose(np.matrix([p2_x, p2_y, 1]))
                    err = p22 - p11_pred
                    if np.linalg.norm(err) < 6:
                        inliers.append(q)
                # whether found best homography
                if len(inliers) > len(good_match)*0.60:
                    best_h = h
                    break
                if len(inliers) > len(max_num):
                    max_num = inliers[:]
                    best_h = h
            # backward wraping image1 based on image2 left_shift
            conor1 = [[0,0],[0,w1-1],[h1-1,0],[h1-1,w1-1]]
            img1_h = []
            img1_w = []
            for p in conor1:
                p2 = np.transpose(np.matrix([p[1], p[0], 1])) # homogeneous representation
                p1 = np.dot(best_h, p2)
                p1 = (1/p1.item(2))*p1
                x = int(p1.item(0))
                y = int(p1.item(1))
                img1_h.append(y)
                img1_w.append(x)
            h1_new = max(img1_h)-min(img1_h)+1
            w1_new = max(img1_w)-min(img1_w)+1
            img_1 = np.zeros((h1_new,w1_new,3), np.uint8)
            for y in range(0, h1):
                for x in range(0, w1):
                    p1 = np.transpose(np.matrix([x, y, 1])) # homogeneous representation
                    p2 = np.dot(best_h, p1)
                    p2 = (1/p2.item(2))*p2
                    xx = int(p2[0])-min(img1_w)
                    yy = int(p2[1])-min(img1_h)
                    img_1[yy][xx] = imgs[i][y][x]
            img_left.append(img_1)
        # right_shift
        for j in range(len(imgs)//2,len(imgs)):
            i = j-1
            h1, w1, c1 = imgs[i].shape
            h2, w2, c2 = imgs[j].shape
            # match two images using features KNN
            D_list = []
            K_list = []
            for f in range(0, len(feature_list[i])):
                f_i = feature_list[i][f]
                d_list = []
                k_list = []
                for ff in range(0, len(feature_list[j])):
                    f_j = feature_list[j][ff]
                    d = np.linalg.norm(f_i - f_j)
                    if len(d_list) < 2:
                        d_list.append(d)
                        k_list.append([kp_list[i][f], kp_list[j][ff]])
                    else:
                        if d < max(d_list):
                            if d_list[0] == max(d_list):
                                d_list.remove(d_list[0])
                                k_list = k_list[1:]
                            else:
                                d_list.remove(d_list[1])
                                k_list = k_list[0:1]
                            d_list.append(d)
                            k_list.append([kp_list[i][f], kp_list[j][ff]])
                D_list.append(d_list)
                K_list.append(k_list)

            # ensure the distance is within a certain ratio of each--Lowe's ratio test
            good_match = []
            for x in range(0, len(D_list)):
                if D_list[x][0] < D_list[x][1]:
                    if D_list[x][0] < 0.75 * D_list[x][1]:
                        good_match.append(K_list[x][0])
                else:
                    if D_list[x][1] < 0.75 * D_list[x][0]:
                        good_match.append(K_list[x][1])
            # RANSAC
            best_h = 0
            max_num = []
            for n in range(1000):
                # ransom find 4 keypoints to calculate homography
                p1 = good_match[random.randrange(0, len(good_match))]
                p2 = good_match[random.randrange(0, len(good_match))]
                p3 = good_match[random.randrange(0, len(good_match))]
                p4 = good_match[random.randrange(0, len(good_match))]
                #print(p)
                p = [p1, p2, p3, p4]
                # caculate homography on four keypoints
                a = []
                for pp in p:
                    p11 = [pp[0][0], pp[0][1], 1]
                    p22 = [pp[1][0], pp[1][1], 1]
                    # SVD the last cotumn v is h
                    # H * p11 = p22
                    a.append([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]])
                    a.append([0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                    #A = np.matric([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]], [0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                A = np.matrix(a)
                u, s, v = np.linalg.svd(A)
                h = np.reshape(v[8], (3, 3))
                h = (1/h.item(8))*h
                # caculate the error between line and points
                inliers = []
                for q in range(0, len(good_match)):
                    p1_x = good_match[q][0][0]
                    p1_y = good_match[q][0][1]
                    p2_x = good_match[q][1][0]
                    p2_y = good_match[q][1][1]
                    p11 = np.transpose(np.matrix([p1_x, p1_y, 1]))
                    p11_pred = np.dot(h, p11)
                    p11_pred = (1/p11_pred.item(2))*p11_pred
                    p22 = np.transpose(np.matrix([p2_x, p2_y, 1]))
                    err = p22 - p11_pred
                    if np.linalg.norm(err) < 6:
                        inliers.append(q)
                # whether found best homography
                if len(inliers) > len(good_match)*0.60:
                    best_h = h
                    break
                if len(inliers) > len(max_num):
                    max_num = inliers[:]
                    best_h = h
            best_hh = best_h.I
            best_hh = (1/best_hh.item(8))*best_hh
            # backward wraping image2 based on image1 right_shift
            conor2 = [[0,0],[0,w2-1],[h2-1,0],[h2-1,w2-1]]
            img2_h = []
            img2_w = []
            for p in conor2:
                p2 = np.transpose(np.matrix([p[1], p[0], 1])) # homogeneous representation
                p1 = np.dot(best_hh, p2)
                p1 = (1/p1.item(2))*p1
                x = int(p1.item(0))
                y = int(p1.item(1))
                img2_h.append(y)
                img2_w.append(x)
            h2_new = max(img2_h)-min(img2_h)+1
            w2_new = max(img2_w)-min(img2_w)+1
            img_2 = np.zeros((h2_new,w2_new,3), np.uint8)
            for y in range(0, h2):
                for x in range(0, w2):
                    p2 = np.transpose(np.matrix([x, y, 1])) # homogeneous representation
                    p1 = np.dot(best_hh, p2)
                    p1 = (1/p1.item(2))*p1
                    xx = int(p1.item(0))-min(img2_w)
                    yy = int(p1.item(1))-min(img2_h)
                    img_2[yy][xx] = imgs[j][y][x]
        img_right.append(img_2)
    else:
        img_mid = imgs[len(imgs)//2]
        # left_shift
        for i in range(0,len(imgs)//2):
            j = i+1
            h1, w1, c1 = imgs[i].shape
            h2, w2, c2 = imgs[j].shape
            # match two images using features KNN
            D_list = []
            K_list = []
            for f in range(0, len(feature_list[i])):
                f_i = feature_list[i][f]
                d_list = []
                k_list = []
                for ff in range(0, len(feature_list[j])):
                    f_j = feature_list[j][ff]
                    d = np.linalg.norm(f_i - f_j)
                    if len(d_list) < 2:
                        d_list.append(d)
                        k_list.append([kp_list[i][f], kp_list[j][ff]])
                    else:
                        if d < max(d_list):
                            if d_list[0] == max(d_list):
                                d_list.remove(d_list[0])
                                k_list = k_list[1:]
                            else:
                                d_list.remove(d_list[1])
                                k_list = k_list[0:1]
                            d_list.append(d)
                            k_list.append([kp_list[i][f], kp_list[j][ff]])
                D_list.append(d_list)
                K_list.append(k_list)

            # ensure the distance is within a certain ratio of each--Lowe's ratio test
            good_match = []
            for x in range(0, len(D_list)):
                if D_list[x][0] < D_list[x][1]:
                    if D_list[x][0] < 0.75 * D_list[x][1]:
                        good_match.append(K_list[x][0])
                else:
                    if D_list[x][1] < 0.75 * D_list[x][0]:
                        good_match.append(K_list[x][1])
            # RANSAC
            best_h = 0
            max_num = []
            for n in range(1000):
                # ransom find 4 keypoints to calculate homography
                p1 = good_match[random.randrange(0, len(good_match))]
                p2 = good_match[random.randrange(0, len(good_match))]
                p3 = good_match[random.randrange(0, len(good_match))]
                p4 = good_match[random.randrange(0, len(good_match))]
                #print(p)
                p = [p1, p2, p3, p4]
                # caculate homography on four keypoints
                a = []
                for pp in p:
                    p11 = [pp[0][0], pp[0][1], 1]
                    p22 = [pp[1][0], pp[1][1], 1]
                    # SVD the last cotumn v is h
                    # H * p11 = p22
                    a.append([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]])
                    a.append([0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                    #A = np.matric([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]], [0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                A = np.matrix(a)
                u, s, v = np.linalg.svd(A)
                h = np.reshape(v[8], (3, 3))
                h = (1/h.item(8))*h
                # caculate the error between line and points
                inliers = []
                for q in range(0, len(good_match)):
                    p1_x = good_match[q][0][0]
                    p1_y = good_match[q][0][1]
                    p2_x = good_match[q][1][0]
                    p2_y = good_match[q][1][1]
                    p11 = np.transpose(np.matrix([p1_x, p1_y, 1]))
                    p11_pred = np.dot(h, p11)
                    p11_pred = (1/p11_pred.item(2))*p11_pred
                    p22 = np.transpose(np.matrix([p2_x, p2_y, 1]))
                    err = p22 - p11_pred
                    if np.linalg.norm(err) < 6:
                        inliers.append(q)
                # whether found best homography
                if len(inliers) > len(good_match)*0.60:
                    best_h = h
                    break
                if len(inliers) > len(max_num):
                    max_num = inliers[:]
                    best_h = h
            # backward wraping image1 based on image2 left_shift
            conor1 = [[0,0],[0,w1-1],[h1-1,0],[h1-1,w1-1]]
            img1_h = []
            img1_w = []
            for p in conor1:
                p2 = np.transpose(np.matrix([p[1], p[0], 1])) # homogeneous representation
                p1 = np.dot(best_h, p2)
                p1 = (1/p1.item(2))*p1
                x = int(p1.item(0))
                y = int(p1.item(1))
                img1_h.append(y)
                img1_w.append(x)
            h1_new = max(img1_h)-min(img1_h)+1
            w1_new = max(img1_w)-min(img1_w)+1
            img_1 = np.zeros((h1_new,w1_new,3), np.uint8)
            for y in range(0, h1):
                for x in range(0, w1):
                    p1 = np.transpose(np.matrix([x, y, 1])) # homogeneous representation
                    p2 = np.dot(best_h, p1)
                    p2 = (1/p2.item(2))*p2
                    xx = int(p2[0])-min(img1_w)
                    yy = int(p2[1])-min(img1_h)
                    img_1[yy][xx] = imgs[i][y][x]
            img_left.append(img_1)
        # right_shift
        for j in range(len(imgs)//2+1,len(imgs)):
            i = j-1
            h1, w1, c1 = imgs[i].shape
            h2, w2, c2 = imgs[j].shape
            # match two images using features KNN
            D_list = []
            K_list = []
            for f in range(0, len(feature_list[i])):
                f_i = feature_list[i][f]
                d_list = []
                k_list = []
                for ff in range(0, len(feature_list[j])):
                    f_j = feature_list[j][ff]
                    d = np.linalg.norm(f_i - f_j)
                    if len(d_list) < 2:
                        d_list.append(d)
                        k_list.append([kp_list[i][f], kp_list[j][ff]])
                    else:
                        if d < max(d_list):
                            if d_list[0] == max(d_list):
                                d_list.remove(d_list[0])
                                k_list = k_list[1:]
                            else:
                                d_list.remove(d_list[1])
                                k_list = k_list[0:1]
                            d_list.append(d)
                            k_list.append([kp_list[i][f], kp_list[j][ff]])
                D_list.append(d_list)
                K_list.append(k_list)

            # ensure the distance is within a certain ratio of each--Lowe's ratio test
            good_match = []
            for x in range(0, len(D_list)):
                if D_list[x][0] < D_list[x][1]:
                    if D_list[x][0] < 0.75 * D_list[x][1]:
                        good_match.append(K_list[x][0])
                else:
                    if D_list[x][1] < 0.75 * D_list[x][0]:
                        good_match.append(K_list[x][1])
            # RANSAC
            best_h = 0
            max_num = []
            for n in range(1000):
                # ransom find 4 keypoints to calculate homography
                p1 = good_match[random.randrange(0, len(good_match))]
                p2 = good_match[random.randrange(0, len(good_match))]
                p3 = good_match[random.randrange(0, len(good_match))]
                p4 = good_match[random.randrange(0, len(good_match))]
                #print(p)
                p = [p1, p2, p3, p4]
                # caculate homography on four keypoints
                a = []
                for pp in p:
                    p11 = [pp[0][0], pp[0][1], 1]
                    p22 = [pp[1][0], pp[1][1], 1]
                    # SVD the last cotumn v is h
                    # H * p11 = p22
                    a.append([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]])
                    a.append([0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                    #A = np.matric([p11[0], p11[1], 1, 0, 0, 0, -p11[0]*p22[0], -p11[1]*p22[0], -p22[0]], [0, 0, 0, p11[0], p11[1], 1, -p11[0]*p22[1], -p11[1]*p22[1], -p22[1]])
                A = np.matrix(a)
                u, s, v = np.linalg.svd(A)
                h = np.reshape(v[8], (3, 3))
                h = (1/h.item(8))*h
                # caculate the error between line and points
                inliers = []
                for q in range(0, len(good_match)):
                    p1_x = good_match[q][0][0]
                    p1_y = good_match[q][0][1]
                    p2_x = good_match[q][1][0]
                    p2_y = good_match[q][1][1]
                    p11 = np.transpose(np.matrix([p1_x, p1_y, 1]))
                    p11_pred = np.dot(h, p11)
                    p11_pred = (1/p11_pred.item(2))*p11_pred
                    p22 = np.transpose(np.matrix([p2_x, p2_y, 1]))
                    err = p22 - p11_pred
                    if np.linalg.norm(err) < 6:
                        inliers.append(q)
                # whether found best homography
                if len(inliers) > len(good_match)*0.60:
                    best_h = h
                    break
                if len(inliers) > len(max_num):
                    max_num = inliers[:]
                    best_h = h
            best_hh = best_h.I
            best_hh = (1/best_hh.item(8))*best_hh
            # backward wraping image2 based on image1 right_shift
            conor2 = [[0,0],[0,w2-1],[h2-1,0],[h2-1,w2-1]]
            img2_h = []
            img2_w = []
            for p in conor2:
                p2 = np.transpose(np.matrix([p[1], p[0], 1])) # homogeneous representation
                p1 = np.dot(best_hh, p2)
                p1 = (1/p1.item(2))*p1
                x = int(p1.item(0))
                y = int(p1.item(1))
                img2_h.append(y)
                img2_w.append(x)
            h2_new = max(img2_h)-min(img2_h)+1
            w2_new = max(img2_w)-min(img2_w)+1
            img_2 = np.zeros((h2_new,w2_new,3), np.uint8)
            for y in range(0, h2):
                for x in range(0, w2):
                    p2 = np.transpose(np.matrix([x, y, 1])) # homogeneous representation
                    p1 = np.dot(best_hh, p2)
                    p1 = (1/p1.item(2))*p1
                    xx = int(p1.item(0))-min(img2_w)
                    yy = int(p1.item(1))-min(img2_h)
                    img_2[yy][xx] = imgs[j][y][x]
            img_right.append(img_2)
    # merge images
    if len(img_left) == 1:
        img_merge = img_left[0]
    else:
        hl, wl, cl = img_left[0].shape
        img_left_resize = cv2.resize(img_left[1], (wl,hl))
        img_merge = copyOver(img_left[0], img_left_resize)
    for i in range(2,len(img_left)):
        hl, wl, cl = img_merge.shape
        img_left_resize = cv2.resize(img_left[i], (wl,hl))
        img_merge = copyOver(img_merge, img_left_resize)
    if len(img_mid) != 0:
        hl, wl, cl = img_merge.shape
        img_mid_resize = cv2.resize(img_mid, (wl,hl))
        #print(img_mid_resize.shape,img_merge.shape)
        img_merge = copyOver(img_merge, img_mid_resize)
    for i in range(0,len(img_right)):
        hl, wl, cl = img_merge.shape
        img_right_resize = cv2.resize(img_right[i], (wl,hl))
        img_merge = copyOver(img_merge, img_right_resize)
    return img_merge

def copyOver(source, destination):
    result_grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(result_grey, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    roi = cv2.bitwise_and(source, source, mask=mask)
    im2 = cv2.bitwise_and(destination, destination, mask=mask_inv)
    result = cv2.add(im2, roi)
    return result


def save_step_4(panoramic_img, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    output_path="./output/step4"
    cv2.imwrite(os.path.join(output_path, 'pano.jpg'), panoramic_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    img_names = []
    for filename in sorted(os.listdir(args.input)):
        print(filename)
        img_names.append(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, img_names, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(imgs)
        save_step_2(modified_imgs, match_list, img_names, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs = up_to_step_3(imgs)
        save_step_3(img_pairs, img_names, args.output)
    elif args.step == 4:
        print("Running step 4")
        panoramic_img = up_to_step_4(imgs)
        save_step_4(panoramic_img, args.output)
