import csv
import math
import numpy
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

dictionary = set()
filename = "train.csv"
train_comment = []
train_topic = []

with open(filename, 'r', encoding='UTF-8') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        del row[:1]

        doc = row[0]
        doc = doc.split()
        trim = []
        for word in doc:
            if word.isalpha():
                dictionary.add(word)
                trim.append(word)

        train_comment.append(trim)

        train_topic.append(row[-1])

filename = "test.csv"
test_comment = []
test_topic = []

with open(filename, 'r', encoding='UTF-8') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        del row[:1]

        doc = row[0]
        doc = doc.split()
        trim = []
        for word in doc:
            if word.isalpha():
                dictionary.add(word)
                trim.append(word)
            elif word[:-1].isalpha():
                dictionary.add(word[:-1])
                trim.append(word[:-1])
            elif word[1:].isalpha():
                dictionary.add(word[1:])
                trim.append(word[1:])


        test_comment.append(trim)

        test_topic.append(row[-1])


dictionary = list(dictionary)


print(len(test_topic))
print(len(train_topic))
print(len(dictionary))

# number of data
N = len(dictionary)
# number of nearest adjacent
K = 2
#number of cluster at the end
C_target = 5000
data = []
# distance between two data
D_original = []
R = []
L = []
G = 2
D_current = []
C_pre = N
C_cur = N // G

def load_data():
    for word in dictionary:
        data.append(word)


def array_distance(a, b):
    if a not in kv.keys():
        return 0.1
    if b not in kv.keys():
        return 0.1
    similar = kv.similarity(a, b)
    dist = 1 / similar
    return dist

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
        A = numpy.array(D_original[i])
        idx = numpy.argpartition(A, K+1)

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



load_data()
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



idf = numpy.zeros(C_target)
cluster_similarity = numpy.zeros((C_target, 3))
# 0 : biology _ 1 : physic _ 2 : chemestry

for i,word in enumerate(dictionary):
    cnt = 0

    for comment in train_comment:
        if word in comment:
            cnt += 1

    for comment in test_comment:
        if word in comment:
            cnt += 1

    cluster_id = L[i]
    tmp = math.log(len(dictionary) / cnt)
    idf[cluster_id] += tmp



for i in range(N):
    cluster_id = L[i]
    tmp1 = kv.similarity(data[i], 'Biology')
    tmp2 = kv.similarity(data[i], 'Physics')
    tmp3 = kv.similarity(data[i], 'Chemistry')
    cluster_similarity[cluster_id][0] += tmp1
    cluster_similarity[cluster_id][1] += tmp2
    cluster_similarity[cluster_id][2] += tmp3

for i in range(C_target):
    count_i = L.count(i)
    idf[i] /= count_i
    cluster_similarity[i][0] /= count_i
    cluster_similarity[i][1] /= count_i
    cluster_similarity[i][2] /= count_i


bag_of_word_vector = []
for i, comment in enumerate(train_comment):
    if i % 200 == 0:
        print(i)


    bag_of_word = numpy.zeros(len(dictionary))
    for word in comment:
        id = dictionary.index(word)
        cluster_id = L[id]
        if train_topic[i] == 'Biology':
            bag_of_word[id] += idf[cluster_id] * cluster_similarity[cluster_id][0]
        elif train_topic[i] == 'Physics':
            bag_of_word[id] += idf[cluster_id] * cluster_similarity[cluster_id][1]
        elif train_topic[i] == 'Chemistry':
            bag_of_word[id] += idf[cluster_id] * cluster_similarity[cluster_id][2]


    bag_of_word_vector.append(bag_of_word)










