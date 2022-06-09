#%%
from tqdm import tqdm
import os

from utils import *
import shutil


#%%
# if __name__ == "__main__":
    # load training data
print("**** loading training labels ****")
train = pd.read_csv("./train.csv")

print("Shape:", train.shape)
print("Unique ID cases:", train["id"].nunique())
print("Missing Values Column:", train.isna().sum().index[-1])
missing_rows = train.isna().sum().values[-1]
print("\t", "with a total missing rows of:", missing_rows)
print("\t", "% of missing rows:", missing_rows / train.shape[0], "\n")

print("Sample of train.csv:")
samples = train.sample(5, random_state=26)
print(samples, "\n")

print("**** get image path ****")
train = get_image_path("./train", train)
samples = train.sample(5, random_state=26)
print(samples, "\n")

#%%
# Sample a few images from specified case
# CASE = "case123"
# sample_paths1 = train[(train["segmentation"].isna() == False) & (train["case"] == CASE)]["image_path"] \
#                     .reset_index().groupby("image_path")["index"].count() \
#                     .reset_index().loc[:9, "image_path"].tolist()

# show_simple_images(sample_paths1, image_names="case123_samples")

# #%%
# # Read image
# path = './train/case131/case131_day0/scans/slice_0067_360_310_1.50_1.50.png'
# img = read_image(path)

# # Get mask
# ID = "case131_day0_slice_0067"
# mask = get_id_mask(train, ID, verbose=False)

# plot_original_mask(img, mask, alpha=1)

# %%
# Create folder to save masks
os.mkdir("./train_masks")

# Get a list of unique ids
unique_ids = train[train["segmentation"].isna()==False]["id"].unique()

for ID in tqdm(unique_ids):
    # Get the mask
    mask = get_id_mask(train, ID, verbose=False)
    # Write it in folder
    cv2.imwrite(f"./train_masks/{ID}.png", mask)

# Save to zip file
# shutil.make_archive('train_masks', 'zip', 'train_masks')

# Delete the initial folder
# shutil.rmtree('masks_png')

# %%
