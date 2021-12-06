import cv2
from PIL import Image # I should need both of these...
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset

# basic transform which converts an image to a tensor
BASIC_TRANSFORM = transforms.ToTensor()

# this function came from https://www.kaggle.com/amalrda/imagesegmentation

def decode_pixels(pix, rows=2100, cols=1400,label=255):
  """Encode the pixel location and length pairs into a mask.
  
  This function is derived from https://www.kaggle.com/amalrda/imagesegmentation (it's basically copied and pasted).
  Take in a row from the 'EncodedPixels' column of the train.csv dataframe and converts it to a mask.
  
  Parameters
  ----------
  pix: list - This is split into pairs of starting pixels and lengths. These correspond to
    an pixels in a 1D array which is later reshaped into the appropriate width and height. 
  
  rows: number of rows the encoded image should be reshaped to. Defaults to 2100.
  
  cols: number of columnss the encoded image should be reshaped to. Defaults to 1400.
  
  label: what value the masked pixels should be assigned. Defaults to 255
  
  Returns
  -------
  np.array - an array that contains the mask of the encoded pixels passed in.
  """
  # if there is information in the pixel list, then parse it and make the (pixel, length) pairs
  if isinstance(pix,str):
    # coverting the string into a list of numbers
    rle_numbers = [int(num_string) for num_string in pix.split(' ')]
    # Coverting them into starting index and length pairs
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
  # otherwise, then make the array of (pixel, length) pairs empty
  else:
    rle_pairs = np.array([])

  # Creating a blank image in form of a single row array
  img = np.zeros(rows*cols, dtype=np.uint8)

  # Setting the segmented pixels in the img
  for ind, length in rle_pairs:
    ind -= 1
    img[ind:ind+length] = label
  img = img.reshape(rows,cols)#,1)
  img = img.T
  return img

class CloudData(Dataset):
  """This class is used for the cloud data"""
  def __init__(self, data_directory = None, mask_df = None, data_type = 'train', transform=BASIC_TRANSFORM, output_width=256, output_height=256):
    """Instantiate the CloudData object.
    
    Arguments
    ---------
    data_directory (str) - path of where the images are saved. Defaults to None.
    
    mask_df (pd.DataFrame) - the dataframe of the training data image names and labels, and encoded pixels. Defaults to None.
    
    data_type (str) - whether the data is of type train or test. If it is test, there is no mask information and only the images are gathered. Defaults to "train".
    
    transform (torch.transforms) - pytorch transform(s) to have applied to the images. Defaults to BASIC_TRANSFORM which simply converts image to tensor.
    
    output_width (int) - the width to have the image tensor and masks outputted at.
    
    output_height (int) - the height to have the image tensor and masks outputted at.
    """
    super(CloudData,self).__init__()
    # the Image_Label field in the mask_df is a string with format {image_name}_{image_label} and need to be broken into two pieces
    mask_df['Label'] = [parts[1] for parts in mask_df['Image_Label'].str.split('_')]
    mask_df['Image'] = [parts[0] for parts in mask_df['Image_Label'].str.split('_')]
    # we are only interested in having one item per unique image and the output for each image will be one mask per class
    self.unique_images = mask_df['Image'].unique()
    self.mask_df = mask_df
    # we need to know the list of unique classes
    self.classes = list(mask_df['Label'].unique())
    self.data_type = data_type
    self.data_directory = data_directory
    self.transform = transform

  def __getitem__(self, idx):
    # get the image name
    image_name = self.unique_images[idx]
    idx_images = self.mask_df[self.mask_df['Image']==image_name]
    # decode the "EncodedPixels" into the mask for each of the classes
    masks = []
    for c in self.classes:
      mask_subset = idx_images[idx_images['Label']==c]
      if mask_subset.shape[0] > 0:
        masks.append(torch.tensor(decode_pixels(mask_subset.iloc[0]['EncodedPixels'], label=1)))
      else:
        masks.append(torch.tensor(decode_pixels(np.nan)))
    
    # get the actual image
    image = Image.open(self.data_directory+'/train_images/'+image_name)
    image_tensor = self.transform(image)
    del image # save memory
    resized_image = F.interpolate(image_tensor.unsqueeze(0), (256, 256))
    resized_mask = F.interpolate(torch.stack(masks).unsqueeze(0), (256, 256))
    return resized_image.squeeze(0).float(), resized_mask.squeeze(0).float()

  def __len__(self):
    return len(self.unique_images)
  
  
