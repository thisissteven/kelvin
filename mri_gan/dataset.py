import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from data_utils.utils import *
from utils import *
from PIL import Image
import cv2
import os
from skimage.metrics import structural_similarity
from utils import *
import os

def get_structural_similarity(image1, image2):
    return structural_similarity(image1, image2, multichannel=True, full=True,
                                 gaussian_weights=True, use_sample_covariance=False, sigma=1.5)

class MRIDataset(Dataset):
    def __init__(self, csv_path, image_folder, transforms=None, frac=1.0, train=True):
        self.transforms = transforms

        self.df = pd.read_csv(csv_path)
        # self.df = self.df.dropna().reset_index(drop=True)
        # self.df = self.df.head(10000)
        self.image_folder = image_folder

        self.fake_df = self.df[self.df['label'] == 'FAKE']
        self.real_df = self.df[self.df['label'] == 'REAL']

        # self.fake_df = fake_df.head(16000)
        # self.real_df = real_df.head(16000)
        self.real_df_len = len(self.real_df)

        self.df = pd.concat([self.fake_df, self.real_df], ignore_index=True)

        # if frac < 1.0:
        #     self.df = self.df.sample(frac=frac).reset_index(drop=True)

        rows_to_remove = []
        self.df['image_path'] = "None"
        self.df['image_path_original'] = "None"

        for idx, row in self.df.iterrows():
          if (row['label'] == "FAKE"):
            path = os.path.join(image_folder, row["videoname"].split(".")[0] + ".jpg")
            path_original = os.path.join(image_folder, row["original"].split(".")[0] + ".jpg")

            if os.path.exists(path) and os.path.exists(path_original):
              self.df.loc[idx, 'image_path'] = path
              self.df.loc[idx, 'image_path_original'] = path_original
            else:
              rows_to_remove.append(idx)
          else:
            path = os.path.join(image_folder, row["videoname"].split(".")[0] + ".jpg")
            if os.path.exists(path):
              self.df.loc[idx, 'image_path'] = path
              self.df.loc[idx, 'image_path_original'] = None
            else:
              rows_to_remove.append(idx)

        self.df = self.df.drop(rows_to_remove).reset_index(drop=True)
        self.df = self.df.drop('original_width', axis=1)
        self.df = self.df.drop('original_height', axis=1)

        # Dataset addition
        dataset_path_new = "/content/drive/MyDrive/DatasetNew/datasetNew"

        for i in range(20):
          rows_to_remove = []

          image_path_new = os.path.join(dataset_path_new, f"DeepFake{str(i).zfill(2)}/DeepFake{str(i).zfill(2)}")
          df_json = pd.read_json(os.path.join(dataset_path_new, f"metadata{i}.json"), orient="index")
          df_json.reset_index(inplace=True)
          df_json.columns = ['videoname', 'label', 'split', 'original']
          df_json = df_json.drop('split', axis=1)
          df_json['image_path'] = "None"
          df_json['image_path_original'] = "None"
          # df_json = df_json.dropna().reset_index(drop=True)

          for idx, row in df_json.iterrows():
            if (row['label'] == 'FAKE'):
              path = os.path.join(image_path_new, row["videoname"].split(".")[0] + ".jpg")
              path_original = os.path.join(image_path_new, row["original"].split(".")[0] + ".jpg")

              if os.path.exists(path) and os.path.exists(path_original):
                df_json.loc[idx, 'image_path'] = path
                df_json.loc[idx, 'image_path_original'] = path_original
              else:
                rows_to_remove.append(idx)
            else:
              path = os.path.join(image_folder, row["videoname"].split(".")[0] + ".jpg")
              if os.path.exists(path):
                df_json.loc[idx, 'image_path'] = path
                df_json.loc[idx, 'image_path_original'] = None
              else:
                rows_to_remove.append(idx)


          df_json = df_json.drop(rows_to_remove).reset_index(drop=True)

          # Concat new data
          self.df = pd.concat([self.df, df_json], ignore_index=True)
        
        # Limit data
        self.fake_df = self.df[self.df['label'] == 'FAKE'].head(16000)
        self.real_df = self.df[self.df['label'] == 'REAL'].head(16000)
        self.df = pd.concat([self.fake_df, self.real_df], ignore_index=True)

        print(self.df)

        self.df_len = len(self.df)
        self.data_dict = self.df.to_dict(orient='records')

        # print(mode)
        # print(f'real ={len(self.fake_df_trimmed)}')
        # print(f'fake ={len(self.real_df)}')
        # print(f'df_len ={self.df_len}')
        # self.df.to_csv('{}_epoch.csv'.format(mode))

    def __getitem__(self, index):
        while True:
            try:
                img_path = self.df.iloc[index]['image_path']
                img_path_original = self.df.iloc[index]['image_path_original']

                res = (256, 256)

                image1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image1_copy = image1.copy()
                image1 = cv2.resize(image1, res, interpolation=cv2.INTER_AREA)
                if (img_path_original != None):
                  image2 = cv2.imread(img_path_original, cv2.IMREAD_COLOR)
                  image2 = cv2.resize(image2, res, interpolation=cv2.INTER_AREA)

                  sim_index, sim = get_structural_similarity(image1, image2)

                  mri_path = ConfigParser.getInstance().get_dfdc_mri_path()
                  mri = 1 - sim
                  mri = (mri * 255).astype(np.uint8)
                else:
                  mri_path = ConfigParser.getInstance().get_dfdc_mri_path()
                  mri = np.zeros((256, 256, 3), np.uint8)
                
                # Make mri dataset
                if mri_path is not None:
                    cv2.imwrite(f"{mri_path}/mri_images/{os.path.basename(img_path)}", mri)
                    cv2.imwrite(f"{mri_path}/images/{os.path.basename(img_path)}", image1_copy)
            
                img_A = Image.open(img_path)
                img_B = Image.fromarray(mri)

                if np.random.random() < 0.5:
                    img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                    img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

                if self.transforms:
                    img_A = self.transforms(img_A)
                    img_B = self.transforms(img_B)

                return {"A": img_A, "B": img_B}

            except Exception as e:
                print(e)
                index = random.randint(0, self.df_len)

    def __len__(self) -> int:
        return self.df_len
