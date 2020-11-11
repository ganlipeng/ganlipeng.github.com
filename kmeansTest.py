import numpy as np
from kmeans import KMeans
import matplotlib
import matplotlib.pyplot as plt

def transform_image(image, code_vectors):

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    
    D1 , D2 , _ = image.shape
    K , _ = code_vectors.shape 

    new_image = np.zeros(image.shape)

    for d1 in range(D1):
        for d2 in range(D2):

            dist = np.zeros(K)
            for k in range(K):
                dist[k] = np.inner(image[d1, d2] - code_vectors[k], image[d1, d2] - code_vectors[k])# 每个点到K个质心的距离

            k = np.argmin(dist)

            new_image[d1, d2] = code_vectors[k] # 每个点赋予最近K的像素值

    return new_image


def kmeans_image_compression():

    im = plt.imread('img2.jpg')
    N, M = im.shape[:2]
    im = im / 255 # 归一化

    data = im.reshape(N * M, 3)

    k_means = KMeans(n_cluster=16, max_iter=100, e=1e-6)
    centroids, _, i = k_means.fit(data)

    print('RGB centroids computed in {} iteration'.format(i))
    new_im = transform_image(im, centroids)

    assert new_im.shape == im.shape, \
        'Shape of transformed image should be same as image'

    mse = np.sum((im - new_im)**2) / (N * M) #原图像与压缩后的图像的均方差
    print('Mean square error per pixel is {}'.format(mse))
    plt.imsave('plots/compressed_cat_16.png', new_im)

    np.savez('plots/k_means_compression_cat_16.npz', im=im, centroids=centroids,
             step=i, new_image=new_im, pixel_error=mse)

if __name__ == '__main__':
    kmeans_image_compression()
