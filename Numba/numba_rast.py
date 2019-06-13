# import array
import numpy as np
from numba import njit, prange
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import time
import sys

# PPM header
w = 512
h = 512
maxval = 255
n = 0
ppm_header = 'P6 ' + str(w) + ' ' + str(h) + ' ' + str(maxval) + '\n'
triangles = []
npTriangles = []
# Write function that creates N triangles and stored into a big array of vertices

# numba cannot update global variables
image = np.array([0, 0, 0] * w * h, dtype=np.ubyte)


def makeTriangles(n, w, h):  # Does not prevent creating duplicate triangles
    # n determines number of triangles within a w x h image to generate in an array

    global triangles, npTriangles
    for i in range(0, n):
        vertices = []
        for j in range(0, 3):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            vertices.append([x, y])  # add one vertex out of 3
        if edgeFunction(vertices[0], vertices[1], vertices[2]) > 0:
            triangles.append(vertices)
        else:
            if i == 0:
                i = 0
            else:
                i = i - 1
    npTriangles = np.array(triangles, dtype=np.float64)
    return


def edgeFunction(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])


@njit  # converts just to LLVM machine code
def main1(tri, param_image):
    global w, h, maxval, ppm_header, triangles, image

    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    c0 = [1, 0, 0]
    c1 = [0, 1, 0]
    c2 = [0, 0, 1]

    # area = edgeFunction(v0, v1, v2)
    area = (v2[0] - v0[0]) * (v1[1] - v0[1]) - (v2[1] - v0[1]) * (v1[0] - v0[0])

    # PPM image data (filled with black)
    # image = array.array('B', [0, 0, 0] * w * h)

    for j in prange(0, h):
        for i in prange(0, w):
            p = [i + 0.5, j + 0.5]
            # w0 = edgeFunction(v1, v2, p)
            w0 = (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0])
            # w1 = edgeFunction(v2, v0, p)
            w1 = (p[0] - v2[0]) * (v0[1] - v2[1]) - (p[1] - v2[1]) * (v0[0] - v2[0])
            # w2 = edgeFunction(v0, v1, p)
            w2 = (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                w0 /= area
                w1 /= area
                w2 /= area
                r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0]
                g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1]
                b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2]
                index = 3 * (j * w + i)
                param_image[index] = int(r * 255)  # red channel
                param_image[index + 1] = int(g * 255)  # green channel
                param_image[index + 2] = int(b * 255)  # blue channel


@njit(parallel=True)  # parallel mode (only works on CPUs but will still run on bender)
def main2(tri, param_image):
    global w, h, maxval, ppm_header, triangles, image

    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    c0 = [1, 0, 0]
    c1 = [0, 1, 0]
    c2 = [0, 0, 1]

    # area = edgeFunction(v0, v1, v2)
    area = (v2[0] - v0[0]) * (v1[1] - v0[1]) - (v2[1] - v0[1]) * (v1[0] - v0[0])

    # PPM image data (filled with black)
    # image = array.array('B', [0, 0, 0] * w * h)

    for j in prange(0, h):
        for i in prange(0, w):
            p = [i + 0.5, j + 0.5]
            # w0 = edgeFunction(v1, v2, p)
            w0 = (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0])
            # w1 = edgeFunction(v2, v0, p)
            w1 = (p[0] - v2[0]) * (v0[1] - v2[1]) - (p[1] - v2[1]) * (v0[0] - v2[0])
            # w2 = edgeFunction(v0, v1, p)
            w2 = (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                w0 /= area
                w1 /= area
                w2 /= area
                r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0]
                g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1]
                b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2]
                index = 3 * (j * w + i)
                param_image[index] = int(r * 255)  # red channel
                param_image[index + 1] = int(g * 255)  # green channel
                param_image[index + 2] = int(b * 255)  # blue channel


def main3(tri, param_image):  # naive mode
    global w, h, maxval, ppm_header, triangles, image

    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    c0 = [1, 0, 0]
    c1 = [0, 1, 0]
    c2 = [0, 0, 1]

    # area = edgeFunction(v0, v1, v2)
    area = (v2[0] - v0[0]) * (v1[1] - v0[1]) - (v2[1] - v0[1]) * (v1[0] - v0[0])

    # PPM image data (filled with black)
    # image = array.array('B', [0, 0, 0] * w * h)

    for j in prange(0, h):
        for i in prange(0, w):
            p = [i + 0.5, j + 0.5]
            # w0 = edgeFunction(v1, v2, p)
            w0 = (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0])
            # w1 = edgeFunction(v2, v0, p)
            w1 = (p[0] - v2[0]) * (v0[1] - v2[1]) - (p[1] - v2[1]) * (v0[0] - v2[0])
            # w2 = edgeFunction(v0, v1, p)
            w2 = (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                w0 /= area
                w1 /= area
                w2 /= area
                r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0]
                g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1]
                b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2]
                index = 3 * (j * w + i)
                param_image[index] = int(r * 255)  # red channel
                param_image[index + 1] = int(g * 255)  # green channel
                param_image[index + 2] = int(b * 255)  # blue channel


choice = input("Enter 1 for @njit, 2 for @njit(Parallel=true), 3 for naive: ")
choice = int(choice)
n = input("Enter number of triangles: ")
makeTriangles(int(n), w, h)

start = time.time()
end = 0
if choice == 1:
    for i in range(0, int(n)):
        main1(npTriangles[i], image)  # image will get continuously updated
    end = time.time()
    end = time.time()
elif choice == 2:
    for i in range(0, int(n)):
        main2(npTriangles[i], image)  # image will get continuously updated
    end = time.time()
elif choice == 3:
    for i in range(0, int(n)):
        main3(npTriangles[i], image)  # image will get continuously updated
    end = time.time()
else:
    print("Invalid choice, exiting program.")
    sys.exit()

with open('temp.ppm', 'wb') as f:
    f.write(bytearray(ppm_header, 'ascii'))
    image.tofile(f)
    im = Image.open("temp.ppm")
    im.save("testpic.png")

print("done")
print("Time elapsed for " + str(n) + " triangles is " + str(end - start))