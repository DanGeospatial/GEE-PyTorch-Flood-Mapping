import rasterio
import matplotlib.pyplot as plt

"""
img_path = "/mnt/d/SAR_testing/images/"

def create_df(path):
    file_name = []
    for dirname, _, filenames in os.walk(path): # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            f = f.replace('tile_', '')
            file_name.append(f)

    return pd.DataFrame({'id': file_name}, index = np.arange(0, len(file_name))).sort_values('id').reset_index(drop=True)


x = create_df(img_path)['id'].values

print(x)
"""
image_1 = "/mnt/d/SAR_Cat/images/tile_0000.png"
image_2 = "/mnt/d/SAR_Cat/images/tile_0001.png"
image_3 = "/mnt/d/SAR_Cat/images/tile_0006.png"

mask_1 = "/mnt/d/SAR_Cat/masks/tile_0000.png"
mask_2 = "/mnt/d/SAR_Cat/masks/tile_0001.png"
mask_3 = "/mnt/d/SAR_Cat/masks/tile_0006.png"

image_1_open = rasterio.open(image_1)
image_2_open = rasterio.open(image_2)
image_3_open = rasterio.open(image_3)

mask_1_open = rasterio.open(mask_1)
mask_2_open = rasterio.open(mask_2)
mask_3_open = rasterio.open(mask_3)


image_1_read = image_1_open.read(1)
image_2_read = image_2_open.read(1)
image_3_read = image_3_open.read(1)

mask_1_read = mask_1_open.read(1)
mask_2_read = mask_2_open.read(1)
mask_3_read = mask_3_open.read(1)


fig = plt.figure(figsize=(6.615, 4.552))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
plt.imshow(image_1_read)
ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                labelleft=False)

ax2 = fig.add_subplot(gs[0, 1])
plt.imshow(image_2_read)
ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                labelleft=False)

ax3 = fig.add_subplot(gs[0, 2])
plt.imshow(image_3_read)
ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                labelleft=False)

ax4 = fig.add_subplot(gs[1, 0])
plt.imshow(mask_1_read, cmap='Greys')
ax4.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                labelleft=False)

ax5 = fig.add_subplot(gs[1, 1])
plt.imshow(mask_2_read, cmap='Greys')
ax5.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                labelleft=False)

ax6 = fig.add_subplot(gs[1, 2])
plt.imshow(mask_3_read, cmap='Greys')
ax6.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                labelleft=False)

plt.tight_layout()
plt.show()


