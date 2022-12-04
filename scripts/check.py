import numpy as np
import cv2
semantic_path_celeb = './assets/recon2/semantic/semantic.png'
# semantic_path_celeb = '/home/jjy/dataset/celeba/test/label/028002.png'
semantic_path_kface = './assets/recon2/semantic.png'

# celeb = cv2.resize(cv2.imread(semantic_path_celeb),(512,512))\
kface = cv2.resize(cv2.imread(semantic_path_kface),(512,512))

canvas_list1,canvas_list2 = [],[]

for i in range(19):
    # canvas1 = np.zeros_like(celeb)
    # canvas1 = np.where(celeb==i,255,0)

    canvas2 = np.zeros_like(kface)
    canvas2 = np.where(kface==i,255,0)

    # canvas_list1.append(canvas1)
    canvas_list2.append(canvas2)

    # cv2.imwrite(f'./scripts/celeba/{str(i).zfill(3)}.png',canvas1)
    # cv2.imwrite(f'./scripts/kface/{str(i).zfill(3)}.png',canvas2)
# canvas_list1_ = np.concatenate(canvas_list1,axis=1)
canvas_list2_ = np.concatenate(canvas_list2,axis=1)
# vis = np.concatenate((canvas_list1_,canvas_list2_),axis=0)

# cv2.imwrite(f'./scripts/check3.png',vis)
cv2.imwrite(f'./scripts/check4.png',canvas_list2_)