import numpy as np
from matplotlib import image
import random
from matplotlib import pyplot



# number of data
N = 410
# number of nearest adjacent
K = 3
#number of cluster at the end
C_target = 41
data = []
# distance between two data
D_original = []
R = []
L = []
G = 2
D_current = []
C_pre = N
C_cur = N // G

def read_images():
    for i in range(N):
        label = i // 10
        im = image.imread(f'ORL\\{i + 1}_{label + 1}.jpg')

        if im.shape == (80, 70, 3):
            R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
            imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            data.append(np.array(imgGray, dtype=np.int64))
        else:
            data.append(np.array(im, dtype=np.int64))


def array_distance(a, b):
    return np.linalg.norm(a-b)

# compute the distance between every 2 data
def initialize_D_original():
    for i in range(N):
        D_original.append([])
        for j in range(N):
            D_original[i].append(array_distance(data[i], data[j]))

# store the k nearest adjacent data for each data
def initialize_R():
    for i in range(N):
        R.append([])
        A = np.array(D_original[i])
        idx = np.argpartition(A, K+1)

        for j in range(K+1):
            R[i].append(idx[j])


# with k nearest data we calculate the distance between 2 data as follow
def initialize_D_current():
    for i in range(N):
        D_current.append([])
        for j in range(N):
            sum = 0
            for x in R[i]:
                for y in R[j]:
                    sum += D_original[x][y]
            avg = sum / ((K+1)**2)
            D_current[i].append(avg)

# every data is in 1 cluster each
def initialize_L():
    for i in range(N):
        L.append(i)


def C_2(n):
    if n == 0 or n == 1:
        return 0
    return (n*(n-1))//2


def rand_index():
    cnt = [0] * C_target
    for i in range(N):
        if L[i] != -1:
            cnt[L[i]] += 1
    print(cnt)
    total = C_2(N)
    TPFP = 0
    for i in range(C_target):
        TPFP += C_2(cnt[i])

    TP = 0
    FN = 0
    for i in range(0, 410, 10):
        l = [0] * C_target
        FNx = C_2(10)
        for j in range(10):
            if L[i+j] != -1:
                l[L[i+j]] += 1

        for j in range(C_target):
            TP += C_2(l[j])
            FNx -= C_2(l[j])

        FN += FNx
    FP = TPFP - TP
    TN = total - (FN + TP + FP)
    print(f'TP = {TP} , TN = {TN} , FP = {FP}, FN = {FN}')
    RI = (TN + TP) / total
    print(RI)

# we update the distance between clusters as follow
def update_D_current():
    for i in range(C_cur):
        Pi = []
        for y in range(N):
            if L[y] == i:
                if y not in Pi:
                    Pi.append(y)
                for q in R[y]:
                    if q not in Pi:
                        Pi.append(q)
        for j in range(C_cur):
            if j == i:
                continue

            Pj = []
            for y in range(N):
                if L[y] == j:
                    if y not in Pj:
                        Pj.append(y)
                    for q in R[y]:
                        if q not in Pj:
                            Pj.append(q)
            sum = 0
            for a in Pi:
                for b in Pj:
                    sum += D_original[a][b]
            avg = sum / (len(Pi) * len(Pj))

            D_current[i][j] = avg


# returns a list of indexes of key element elected by the given heuristic
def find_key_element(number_of_key):
    key = []
    first_key = -1
    mini = 1000000000000000000

    for i in range(C_pre):
        sum = 0
        for j in range(C_pre):
            sum += D_current[i][j]

        if mini > sum:
            mini = sum
            first_key = i

    key.append(first_key)
    while len(key) < number_of_key:
        maxi = 0
        id = -1
        for i in range(C_pre):
            mini = 1000000000000000000000
            if i not in key:
                for k in key:
                    if D_current[i][k] < mini:
                        mini = D_current[i][k]

                if mini > maxi:
                    maxi = mini
                    id = i

        key.append(id)

    return key


def merge_cluster(key):
    for i in range(C_pre):
        # if an element is not in key element
        if i not in key:
            mini = 1000000000000000000
            id = -1
            # find the nearest key element
            for k in key:

                if D_current[i][k] < mini:
                    mini = D_current[i][k]
                    id = k

            # change the cluster number of the element into key cluster id
            for x in range(N):
                if L[x] == i:
                    L[x] = id


# we need to assure that for every i -> L[i] is one of these {0, 1, 2, 3, ... , C_cur}
# so we reassign the cluster id
def re_assign():
    mark = [0] * N
    id = 0
    for color in range(C_cur):

        for i in range(N):
            if mark[i] == 0:
                id = L[i]
                break

        for i in range(N):
            if mark[i] == 0 and L[i] == id:
                L[i] = color
                mark[i] = 1



read_images()
initialize_D_original()
initialize_R()
initialize_D_current()
initialize_L()




while C_cur > C_target:
    key = find_key_element(C_cur)
    merge_cluster(key)
    re_assign()
    update_D_current()
    C_pre = C_cur
    C_cur = C_cur // G


C_cur = C_target
key = find_key_element(C_cur)
merge_cluster(key)
re_assign()
update_D_current()

rand_index()