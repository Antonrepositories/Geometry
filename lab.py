from functools import cmp_to_key
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Circle
import math
from math import sqrt
from random import randint, shuffle
mid = [0, 0]

def quad(p):
    if p[0] >= 0 and p[1] >= 0:
        return 1
    if p[0] <= 0 and p[1] >= 0:
        return 2
    if p[0] <= 0 and p[1] <= 0:
        return 3
    return 4

def orientation(a, b, c):
    res = (b[1]-a[1]) * (c[0]-b[0]) - (c[1]-b[1]) * (b[0]-a[0])
    if res == 0:
        return 0
    if res > 0:
        return 1
    return -1

def compare(p1, q1):
    p = [p1[0]-mid[0], p1[1]-mid[1]]
    q = [q1[0]-mid[0], q1[1]-mid[1]]
    one = quad(p)
    two = quad(q)
    if one != two:
        if one < two:
            return -1
        return 1
    if p[1]*q[0] < q[1]*p[0]:
        return -1
    return 1


def merger(a, b):
    n1, n2 = len(a), len(b)
    ia, ib = 0, 0
    for i in range(1, n1):
        if a[i][0] > a[ia][0]:
            ia = i
    for i in range(1, n2):
        if b[i][0] < b[ib][0]:
            ib = i
    inda, indb = ia, ib
    done = 0
    while not done:
        done = 1
        while orientation(b[indb], a[inda], a[(inda+1) % n1]) >= 0:
            inda = (inda + 1) % n1
        while orientation(a[inda], b[indb], b[(n2+indb-1) % n2]) <= 0:
            indb = (indb - 1) % n2
            done = 0
    uppera, upperb = inda, indb
    inda, indb = ia, ib
    done = 0
    g = 0
    while not done:
        done = 1
        while orientation(a[inda], b[indb], b[(indb+1) % n2]) >= 0:
            indb = (indb + 1) % n2
        while orientation(b[indb], a[inda], a[(n1+inda-1) % n1]) <= 0:
            inda = (inda - 1) % n1
            done = 0
    ret = []
    lowera, lowerb = inda, indb
    ind = uppera
    ret.append(a[uppera])
    while ind != lowera:
        ind = (ind+1) % n1
        ret.append(a[ind])
    ind = lowerb
    ret.append(b[lowerb])
    while ind != upperb:
        ind = (ind+1) % n2
        ret.append(b[ind])
    return ret

def bruteHull(a):
    global mid
    s = set()
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            x1, x2 = a[i][0], a[j][0]
            y1, y2 = a[i][1], a[j][1]
            a1, b1, c1 = y1-y2, x2-x1, x1*y2-y1*x2
            pos, neg = 0, 0
            for k in range(len(a)):
                if (k == i) or (k == j) or (a1*a[k][0]+b1*a[k][1]+c1 <= 0):
                        neg += 1
                if (k == i) or (k == j) or (a1*a[k][0]+b1*a[k][1]+c1 >= 0):
                        pos += 1
            if pos == len(a) or neg == len(a):
                s.add(tuple(a[i]))
                s.add(tuple(a[j]))
    ret = []
    for x in s:
        ret.append(list(x))
    mid = [0, 0]
    n = len(ret)
    for i in range(n):
        mid[0] += ret[i][0]
        mid[1] += ret[i][1]
        ret[i][0] *= n
        ret[i][1] *= n
    ret = sorted(ret, key=cmp_to_key(compare))
    for i in range(n):
        ret[i] = [ret[i][0]/n, ret[i][1]/n]
    return ret

def divide(a):
    if len(a) <= 5:
        return bruteHull(a)
    left, right = [], []
    start = int(len(a)/2)
    for i in range(start):
        left.append(a[i])
    for i in range(start, len(a)):
        right.append(a[i])
    left_hull = divide(left)
    right_hull = divide(right)
    return merger(left_hull, right_hull)


def generate_random_points(n):
    points = []
    for _ in range(n):
        x = random.uniform(-10000, 10000) * random.uniform(0,10)  # Від 0 до 5
        y = random.uniform(-10000, 10000) * random.uniform(0, 10)  # Від 0 до 5
        points.append([x, y])
    return points

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
INF = 1e18

def dist(a, b):
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

def is_inside(c, p):
    return dist(c, p) <= c[2]

def get_circle_center(bx, by, cx, cy):
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    return [(cy * B - by * C) / (2 * D), (bx * C - cx * B) / (2 * D)]

def circle_from1(A, B):
    C = [(A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0]
    return C + [dist(A, B) / 2.0]

def circle_from2(A, B, C):
    I = get_circle_center(B[0] - A[0], B[1] - A[1], C[0] - A[0], C[1] - A[1])
    I[0] += A[0]
    I[1] += A[1]
    return I + [dist(I, A)]

def is_valid_circle(c, P):
    for p in P:
        if not is_inside(c, p):
            return False
    return True

def min_circle_trivial(P):
    assert(len(P) <= 3)
    if not P:
        return [0, 0, 0]
    elif len(P) == 1:
        return P[0] + [0]
    elif len(P) == 2:
        return circle_from1(P[0], P[1])
    for i in range(3):
        for j in range(i + 1, 3):
            c = circle_from1(P[i], P[j])
            if is_valid_circle(c, P):
                return c
    return circle_from2(P[0], P[1], P[2])

def welzl_helper(P, R, n):
    if n == 0 or len(R) == 3:
        return min_circle_trivial(R)
    idx = randint(0, n - 1)
    p = P[idx]
    P[idx], P[n - 1] = P[n - 1], P[idx]
    d = welzl_helper(P, R.copy(), n - 1)
    if is_inside(d, p):
        return d
    R.append(p)
    return welzl_helper(P, R.copy(), n - 1)

def welzl(P):
    P_copy = P.copy()
    shuffle(P_copy)
    return welzl_helper(P_copy, [], len(P_copy))

 
# Driver Code
if __name__ == '__main__':
    a = []
    q = int(input("Enter amout of points to generate: "))
    a = generate_random_points(q)
    n = len(a)
    a.sort()
    ans = divide(a)
    print('Convex Hull:')
    for x in ans:
        print(int(x[0]), int(x[1]))
    #print(ans)
    x_coords = [point[0] for point in a]
    y_coords = [point[1] for point in a]
    ans.append(ans[0])
    connected_x_coords = [point[0] for point in ans]
    connected_y_coords = [point[1] for point in ans]
    ans.pop()
    min_circle = welzl(ans)
    # Малюємо результат
    fig, ax = plt.subplots()
    center = [min_circle[0], min_circle[1]]
    radius = min_circle[2]
    circle_patch = Circle(center, radius, edgecolor='green', facecolor='none')
    ax.add_patch(circle_patch)
    plt.scatter(x_coords, y_coords, color='blue')
    plt.plot(connected_x_coords, connected_y_coords, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Points')
    plt.grid(True)
    plt.show()


