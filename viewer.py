import matplotlib.pyplot as plt
img = opener.open('dim2,porsty0.3,blobns2,noise0.05,angles180_test')['processed']

_, ax = plt.subplots(figsize = (7,7))
ax.imshow(img, cmap = 'gray')
plt.show()