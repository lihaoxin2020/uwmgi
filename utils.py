import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import torch
from tqdm import tqdm
import glob


# ref: https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration?scriptVersionId=96824919&cellId=42
def mask_from_segmentation(segmentation, shape):
    """Returns the mask corresponding to the input segmentation.
    segmentation: a list of start points and lengths in this order
    max_shape: the shape to be taken by the mask
    return:: a 2D mask"""

    # Get a list of numbers from the initial segmentation
    segm = np.asarray(segmentation.split(), dtype=int)

    # Get start point and length between points
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Compute the location of each endpoint
    end_point = start_point + length_point

    # Create an empty list mask the size of the original image
    # take onl
    case_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Change pixels from 0 to 1 that are within the segmentation
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = 255

    case_mask = case_mask.reshape((shape[0], shape[1]))

    return case_mask


# ref: https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration?scriptVersionId=96824919&cellId=37
def read_image(path):
    """Reads and converts the image.
    path: the full complete path to the .png file"""

    # Read image in a corresponding manner
    # convert int16 -> float32
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    # Scale to [0, 255]
    image = cv2.normalize(image, None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image.astype(np.uint8)

    return image


# ref: https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration?scriptVersionId=96824919&cellId=38
def show_simple_images(sample_paths, image_names="sample_images"):
    """Displays simple images (without mask)."""

    # Get additional info from the path
    case_name = [info.split("_")[0][-7:] for info in sample_paths]
    day_name = [info.split("_")[1].split("/")[0] for info in sample_paths]
    slice_name = [info.split("_")[2] for info in sample_paths]


    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(23, 8))
    axs = axs.flatten()

    for k, path in enumerate(sample_paths):
        title = f"{k+1}. {case_name[k]} - {day_name[k]} - {slice_name[k]}"
        axs[k].set_title(title, fontsize=14, weight='bold')

        img = read_image(path)
        axs[k].imshow(img)
        axs[k].axis("off")

    plt.tight_layout()
    plt.show()


# ref: https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration?scriptVersionId=96824919&cellId=48
def plot_original_mask(img, mask, alpha=1):
    def CustomCmap(rgb_color):

        r1, g1, b1 = rgb_color

        cdict = {'red': ((0, r1, r1),
                        (1, r1, r1)),
                'green': ((0, g1, g1),
                        (1, g1, g1)),
                'blue': ((0, b1, b1),
                        (1, b1, b1))}

        cmap = LinearSegmentedColormap('custom_cmap', cdict)
        return cmap

    # --- Custom Color Maps ---
    # Yellow Purple Red
    mask_colors = [(1.0, 0.7, 0.1), (1.0, 0.5, 1.0), (1.0, 0.22, 0.099)]
    legend_colors = [Rectangle((0,0),1,1, color=color) for color in mask_colors]
    labels = ["Large Bowel", "Small Bowel", "Stomach"]

    CMAP1 = CustomCmap(mask_colors[0])
    CMAP2 = CustomCmap(mask_colors[1])
    CMAP3 = CustomCmap(mask_colors[2])

    # Change pixels - when 1 make True, when 0 make NA
    mask = np.ma.masked_where(mask == 0, mask)

    # Split the channels
    mask_largeB = mask[:, :, 0]
    mask_smallB = mask[:, :, 1]
    mask_stomach = mask[:, :, 2]

    # Plot the 2 images (Original and with Mask)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Original
    ax1.set_title("Original Image")
    ax1.imshow(img)
    ax1.axis("off")

    # With Mask
    ax2.set_title("Image with Mask")
    ax2.imshow(img)
    ax2.imshow(mask_largeB, interpolation='none', cmap=CMAP1, alpha=alpha)
    ax2.imshow(mask_smallB, interpolation='none', cmap=CMAP2, alpha=alpha)
    ax2.imshow(mask_stomach, interpolation='none', cmap=CMAP3, alpha=alpha)
    ax2.legend(legend_colors, labels)
    ax2.axis("off")

    plt.show()


def get_img_size(x, flag):
    if x != 0:
        split = x.split("_")
        width = split[3]
        height = split[4]

        if flag == "width":
            return int(width)
        elif flag == "height":
            return int(height)

    return 0


def get_pixel_size(x, flag):
    if x != 0:
        split = x.split("_")
        width = split[-2]
        height = ".".join(split[-1].split(".")[:-1])

        if flag == "width":
            return float(width)
        elif flag == "height":
            return float(height)

    return 0


# ref: Reference: https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration?scriptVersionId=96824919&cellId=18
def get_image_path(base_path, df):
    """Gets the case, day, slice_no and path of the dataset (either train or test).
    base_path: path to train image folder. 
    return :: modified dataframe"""

    # Create case, day and slice columns
    df["case"] = df["id"].apply(lambda x: x.split("_")[0])
    df["day"] = df["id"].apply(lambda x: x.split("_")[1])
    df["slice_no"] = df["id"].apply(lambda x: x.split("_")[-1])

    df["image_path"] = 0

    n = len(df)

    # Loop through entire dataset
    # for k in tqdm(range(n)):
    #     data = df.iloc[k, :]

    #     # # In case coordinates for healthy tissue are present
    #     # if not pd.isnull(df.iloc[k, 2]):
    #     case = data.case
    #     day = data.day
    #     slice_no = data.slice_no
    #     # Change value to the correct one
    #     df.loc[k, "image_path"] = glob.glob(f"{base_path}/{case}/{case}_{day}/scans/slice_{slice_no}*")[0]

    def image_path(ID):
        case = ID.split("_")[0]
        day = ID.split("_")[1]
        slice_no = ID.split("_")[-1]
        return glob.glob(f"{base_path}/{case}/{case}_{day}/scans/slice_{slice_no}*")[0]

    df["image_path"] = df.id.map(image_path)

    df["image_width"] = df["image_path"].apply(lambda x: get_img_size(x, "width"))
    df["image_height"] = df["image_path"].apply(lambda x: get_img_size(x, "height"))
    df["pixel_width"] = df["image_path"].apply(lambda x: get_pixel_size(x, "width"))
    df["pixel_height"] = df["image_path"].apply(lambda x: get_pixel_size(x, "height"))

    return df


# ref: https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration?scriptVersionId=96824919&cellId=45
def get_id_mask(train, ID, verbose=False):
    """Returns a mask for each case ID. If no segmentation was found, the mask will be empty
    - meaning formed by only 0
    ID: the case ID from the train.csv file
    verbose: True if we want any prints
    return: segmentation mask"""

    # ~~~ Get the data ~~~
    # Get the portion of dataframe where we have ONLY the speciffied ID
    ID_data = train[train["id"] == ID].reset_index(drop=True)

    # Split the dataframe into 3 series of observations
    # each for one speciffic class - "large_bowel", "small_bowel", "stomach"
    observations = [ID_data.loc[k, :] for k in range(3)]

    # ~~~ Create the mask ~~~
    # Get the maximum height out of all observations
    # if max == 0 then no class has a segmentation
    # otherwise we keep the length of the mask
    max_height = np.max([obs.image_height for obs in observations])
    max_width = np.max([obs.image_width for obs in observations])

    # Get shape of the image
    # 3 channels of color/classes
    shape = (max_height, max_width, 3)

    # Create an empty mask with the shape of the image
    mask = np.zeros(shape, dtype=np.uint8)

    # If there is at least 1 segmentation found in the group of 3 classes
    if max_height != 0:
        for k, location in enumerate(["large_bowel", "small_bowel", "stomach"]):
            observation = observations[k]
            segmentation = observation.segmentation

            # If a segmentation is found
            # Append a new channel to the mask
            if not pd.isnull(segmentation):
                mask[..., k] = mask_from_segmentation(segmentation, shape)

    # If no segmentation was found skip
    elif max_height == 0:
        mask = None
        if verbose:
            print("None of the classes have segmentation.")

    return mask


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    img = img.astype('float32')  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_msk(path):
    if not os.path.exists(path):
        return None
    msk = cv2.imread(path)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
