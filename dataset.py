import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from utils import *


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if msk is None:
                msk = np.zeros_like(img)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


if __name__ == '__main__':
    class CFG:
        seed = 42
        debug = False  # set debug=False for Full Training
        exp_name = 'Baselinev2'
        comment = 'unet-efficientnet_b1-224x224-aug2-split2'
        model_name = 'Unet'
        backbone = 'efficientnet-b1'
        train_bs = 1
        valid_bs = train_bs * 2
        img_size = [224, 224]
        epochs = 15
        lr = 2e-3
        scheduler = 'CosineAnnealingLR'
        min_lr = 1e-6
        T_max = int(30000 / train_bs * epochs) + 50
        T_0 = 25
        warmup_epochs = 0
        wd = 1e-6
        n_accumulate = max(1, 32 // train_bs)
        n_fold = 5
        num_classes = 3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_transforms = {
        "train": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0] // 20, max_width=CFG.img_size[1] // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }

    df = pd.read_csv('./train.csv')
    df = get_image_path("./train", df.head(1000))
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len)  # length of each rle mask
    # df['mask_path'] = df.mask_path.str.replace('/png/', '/np').str.replace('.png', '.npy')

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()  # rle list of each id
    df2 = df2.merge(
        df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())  # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len == 0)  # empty masks

    def mask_path(ID):
        return f"./train_masks/{ID}.png"

    df['mask_path'] = df.id.map(mask_path)

    train_dataset = BuildDataset(df, transforms=data_transforms['train'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)

    def pthtensor2ndarray(image):
        return np.transpose(image.numpy(), (1, 2, 0))

    for idx, item in enumerate(train_loader):
        images = item[0]
        masks = item[1]

        for i in range(CFG.train_bs):
            if masks.sum() != 0:
                plot_original_mask(pthtensor2ndarray(images[i]), pthtensor2ndarray(masks[i]))

    print("done!")

