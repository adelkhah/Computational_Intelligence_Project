import numpy as np
from matplotlib import image
import random
from matplotlib import pyplot

class Our_clsuter_af():

    def __init__(self, N, K, C_target, G, data) -> None:
        self.N = N                  # number of data
        self.K = K                  # number of nearest adjacent
        self.C_target = C_target    # number of cluster at the end
        self.G = G
        self.C_pre = N
        self.C_cur = N // G
        self.data = data

        self.D_original = []
        self.R = []
        self.L = []
        self.D_current = []

    # number of data
    # N = 410
    # number of nearest adjacent
    # K = 3
    #number of cluster at the end
    # C_target = 41
    # data = []
    # distance between two data
    # D_original = []
    # R = []
    # L = []
    # G = 2
    # D_current = []
    # C_pre = N
    # C_cur = N // G

    def get_L(self):
        return self.L

    def read_images(self):
        for i in range(self.N):
            label = i // 10
            im = image.imread(f'./ORL/{i + 1}_{label + 1}.jpg')

            if im.shape == (80, 70, 3):
                R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
                imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
                self.data.append(np.array(imgGray, dtype=np.int64))
            else:
                self.data.append(np.array(im, dtype=np.int64))


    def array_distance(self, a, b):
        return np.linalg.norm(a-b)

    # compute the distance between every 2 data
    def initialize_D_original(self):
        for i in range(self.N):
            self.D_original.append([])
            for j in range(self.N):
                self.D_original[i].append(self.array_distance(self.data[i], self.data[j]))

    # store the k nearest adjacent data for each data
    def initialize_R(self):
        for i in range(self.N):
            self.R.append([])
            A = np.array(self.D_original[i])
            idx = np.argpartition(A, self.K+1)

            for j in range(self.K+1):
                self.R[i].append(idx[j])


    # with k nearest data we calculate the distance between 2 data as follow
    def initialize_D_current(self):
        for i in range(self.N):
            self.D_current.append([])
            for j in range(self.N):
                sum = 0
                for x in self.R[i]:
                    for y in self.R[j]:
                        # print(x, y)
                        sum += self.D_original[x][y]
                avg = sum / ((self.K+1)**2)
                self.D_current[i].append(avg)

    # every data is in 1 cluster each
    def initialize_L(self):
        for i in range(self.N):
            self.L.append(i)


    def C_2(self, n):
        if n == 0 or n == 1:
            return 0
        return (n*(n-1))//2


    def rand_index(self):
        cnt = [0] * self.C_target
        for i in range(self.N):
            # print(i)
            if self.L[i] != -1:
                cnt[self.L[i]] += 1
        # print(cnt)
        total = self.C_2(self.N)
        TPFP = 0
        for i in range(self.C_target):
            TPFP += self.C_2(cnt[i])

        TP = 0
        FN = 0
        for i in range(0, self.N, 10):
            l = [0] * self.C_target
            FNx = self.C_2(10)
            for j in range(10):
                if self.L[i+j] != -1:
                    l[self.L[i+j]] += 1

            for j in range(self.C_target):
                TP += self.C_2(l[j])
                FNx -= self.C_2(l[j])

            FN += FNx
        FP = TPFP - TP
        TN = total - (FN + TP + FP)
        # print()
        RI = (TN + TP) / total
        # print(RI)
        return RI, f'TP = {TP} , TN = {TN} , FP = {FP}, FN = {FN}'

    # we update the distance between clusters as follow
    def update_D_current(self):
        for i in range(self.C_cur):
            Pi = []
            for y in range(self.N):
                if self.L[y] == i:
                    if y not in Pi:
                        Pi.append(y)
                    for q in self.R[y]:
                        if q not in Pi:
                            Pi.append(q)
            for j in range(self.C_cur):
                if j == i:
                    continue

                Pj = []
                for y in range(self.N):
                    if self.L[y] == j:
                        if y not in Pj:
                            Pj.append(y)
                        for q in self.R[y]:
                            if q not in Pj:
                                Pj.append(q)
                sum = 0
                for a in Pi:
                    for b in Pj:
                        sum += self.D_original[a][b]
                avg = sum / (len(Pi) * len(Pj))

                self.D_current[i][j] = avg


    # returns a list of indexes of key element elected by the given heuristic
    def find_key_element(self, number_of_key):
        key = []
        first_key = -1
        mini = 1000000000000000000

        for i in range(self.C_pre):
            sum = 0
            for j in range(self.C_pre):
                sum += self.D_current[i][j]

            if mini > sum:
                mini = sum
                first_key = i

        key.append(first_key)
        while len(key) < number_of_key:
            maxi = 0
            id = -1
            for i in range(self.C_pre):
                mini = 1000000000000000000000
                if i not in key:
                    for k in key:
                        if self.D_current[i][k] < mini:
                            mini = self.D_current[i][k]

                    if mini > maxi:
                        maxi = mini
                        id = i

            key.append(id)

        return key


    def merge_cluster(self, key):
        for i in range(self.C_pre):
            # if an element is not in key element
            if i not in key:
                mini = 1000000000000000000
                id = -1
                # find the nearest key element
                for k in key:

                    if self.D_current[i][k] < mini:
                        mini = self.D_current[i][k]
                        id = k

                # change the cluster number of the element into key cluster id
                for x in range(self.N):
                    if self.L[x] == i:
                        self.L[x] = id


    # we need to assure that for every i -> L[i] is one of these {0, 1, 2, 3, ... , C_cur}
    # so we reassign the cluster id
    def re_assign(self):
        mark = [0] * self.N
        id = 0
        for color in range(self.C_cur):

            for i in range(self.N):
                if mark[i] == 0:
                    id = self.L[i]
                    break

            for i in range(self.N):
                if mark[i] == 0 and self.L[i] == id:
                    self.L[i] = color
                    mark[i] = 1



    # read_images()
    



    def run_algorithm(self):
        self.initialize_D_original()
        self.initialize_R()
        self.initialize_D_current()
        self.initialize_L()

        while self.C_cur > self.C_target:
            key = self.find_key_element(self.C_cur)
            self.merge_cluster(key)
            self.re_assign()
            self.update_D_current()
            self.C_pre = self.C_cur
            self.C_cur = self.C_cur // self.G


        self.C_cur = self.C_target
        self.key = self.find_key_element(self.C_cur)
        self.merge_cluster(key)
        self.re_assign()
        self.update_D_current()

        # self.rand_index()