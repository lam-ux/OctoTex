import glob
import os

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
PATH_CHK = "checkpoints/rough/rough_net_last.pth"
CROP = 1024

# %%
transform = transforms.Compose([
    transforms.Resize(CROP),
    transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (input - mean) / std
    # outputs range from -1 to 1
])

transformDoNotResize = transforms.Compose([
    # transforms.Resize(CROP),
    # transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (input - mean) / std
    # outputs range from -1 to 1
])


class TestDataset(Dataset):
    def __init__(self, img_dir, single=False):
        if (single):
            self.file_list = glob.glob(img_dir)
            self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]
            return

        self.file_list = glob.glob(img_dir + "/*.png")
        self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.file_list[i]).convert('RGB')
        h, w = img.size

        if w < 256 or h < 256 or w - 300 > h or h - 300 > w or w > 2048 or h > 2048:
            img = transform(img)
        else:
            img = transformDoNotResize(img)

        return img, self.names[i]


# %% test
def generateRough(net, DIR_FROM, DIR_EVAL):
    output_normal = DIR_EVAL
    if not os.path.exists(output_normal):
        os.makedirs(output_normal)

    data_test = TestDataset(DIR_FROM)
    # print(batch_size)
    testloader = DataLoader(data_test, batch_size=1, shuffle=False)

    print("\nOutput disp files...")

    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            img_in = data[0].to(device)
            img_out = net(img_in)
            # print(img_name)

            img_out_filename = os.path.join(output_normal, f"{data[1][0]}_rough.png")
            save_image(img_out, img_out_filename, value_range=(-1, 1), normalize=True)

            im = Image.open(img_out_filename).convert("L")
            bright = ImageEnhance.Brightness(im)

            factor = 0.7
            im_output = bright.enhance(factor)
            sharp = ImageEnhance.Contrast(im_output)

            factor = 1.5
            im_output = sharp.enhance(factor)
            im_output.save(img_out_filename)

    print("Done!")


def generateRoughSingle(net, DIR_FROM, DIR_EVAL):
    output_normal = DIR_EVAL
    if not os.path.exists(output_normal):
        os.makedirs(output_normal)

    data_test = TestDataset(DIR_FROM, True)
    # print(batch_size)
    testloader = DataLoader(data_test, batch_size=1, shuffle=False)

    print("\nOutput disp files...")

    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            img_in = data[0].to(device)
            img_out = net(img_in)
            # print(img_name)

            img_out_filename = os.path.join(output_normal, f"{data[1][0]}_rough.png")
            save_image(img_out, img_out_filename, value_range=(-1, 1), normalize=True)

            im = Image.open(img_out_filename).convert("L")
            enhancer = ImageEnhance.Contrast(im)

            factor = 1.1
            im_output = enhancer.enhance(factor)
            im_output.save(img_out_filename)

    print("Done!")


if __name__ == "__main__":
    from model import PBR

    norm_net = PBR(8).to(device)
    checkpoint = torch.load(PATH_CHK)
    norm_net.load_state_dict(checkpoint)

    generateRough(norm_net, "textures", "out")
