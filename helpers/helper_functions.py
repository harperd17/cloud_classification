import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
from torchsat.transforms import transforms_seg
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


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
  def __init__(self, data_directory = None, mask_df = None, data_type = 'train', transform=None, output_width=256, output_height=256, normalize_func = None, preprocessing=None):
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
    self.normalize_func = normalize_func
    self.preprocessing = preprocessing

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
    image_tensor = torchvision.transforms.ToTensor()(image)
    del image # save memory
    resized_image = (F.interpolate(image_tensor.unsqueeze(0), (output_width, output_height))).squeeze(0).float()
    resized_mask = (F.interpolate(torch.stack(masks).unsqueeze(0), (output_width, output_height))).squeeze(0).float()
    if self.preprocessing:
      preprocessed = self.preprocessing(image=resized_image, mask=resized_mask)
      resized_img = preprocessed['image']
      resized_mask = preprocessed['mask']
    if self.transform is None:
      if self.normalize_func is None:
        return resized_image, resized_mask
      else:
        return_img, return_mask = self.normalize_func(resized_image, resized_mask)
        return return_img, return_mask.float()
    else:
      if self.normalize_func is None:
        return_img, return_mask = self.perform_transform(resized_image, resized_mask,self.transform)
        return return_img, return_mask.float()
      else:
        return_img, return_mask = self.normalize_func(self.perform_transform(resized_image,resized_mask,self.transform))
        return return_img, return_mask.float()

    #return resized_image.squeeze(0).float(), resized_mask.squeeze(0).float()

  def __len__(self):
    return len(self.unique_images)

  def perform_transform(self,img, mask, transform_list):
    img = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
    mask = (mask.permute(1,2,0).numpy()*255).astype(np.uint8)
    transformed_img, transformed_mask = transforms_seg.Compose(
        transform_list
    )(img, mask)                                                                                               
    return torch.tensor(transformed_img/255,dtype=torch.float32).permute(2,0,1), torch.tensor(transformed_mask/255,dtype=torch.float32).permute(2,0,1)
    
  
def show_image_and_masks(images_and_masks, class_labels = None, inches_per_image = 3):
  total_sets = len(images_and_masks)
  images_per_set = 1 + images_and_masks[0][1].shape[0]
  fig, ax = plt.subplots(total_sets,images_per_set)
  fig.set_size_inches((inches_per_image*images_per_set,inches_per_image*total_sets))
  for i, set_of_images in enumerate(images_and_masks):
    ax[i][0].imshow(set_of_images[0].permute(1,2,0))
    ax[i][0].axis('off')
    for j in range(1,images_per_set):
      ax[i][j].imshow(set_of_images[1][j-1,:,:])
      ax[i][j].axis('off')
  # now go through each class and add it as a title in the first row of masks
  ax[0][0].set_title('Raw Image')
  for j in range(1,images_per_set):
    ax[0][j].set_title(class_labels[j-1])
  fig.tight_layout()
  return fig, ax


def show_predicted_masks(pred_masks,true_masks, original_images, inches_per_img=3, classes = None, cmap='coolwarm'):
  sigmoid_layer = nn.Sigmoid()
  # first, check to make sure there are the same number of pred and true masks
  if len(pred_masks) != len(true_masks):
    raise ValueError(f"There must be the same number of pred masks as true masks. There are {len(pred_masks)} pred masks and {len(true_masks)} true masks.")
  rows = len(pred_masks)*2
  columns = pred_masks[0].shape[0]
  fig, ax = plt.subplots(rows,columns+1)
  fig.set_size_inches((columns*inches_per_img,rows*inches_per_img))
  for r in range(int(rows/2)):
    for c in range(columns):
      ax[r*2][c+1].imshow(true_masks[r][c,:,:])
      im = ax[r*2+1][c+1].imshow(sigmoid_layer(pred_masks[r][c,:,:]),cmap=cmap)
      if not cmap is None:
        fig.colorbar(im,ax=ax[r*2+1][c+1])
      if c == 0:
        ax[r*2][c].set_ylabel('True Masks')
        ax[r*2+1][c].set_ylabel('Pred Masks')
    ax[r*2][0].imshow(original_images[r])
    ax[r*2+1][0].imshow(original_images[r])
  if not classes is None:
    for c in range(columns):
      ax[0][c+1].set_title(classes[c])
  fig.tight_layout()
  return fig, ax

"""This function is copied from https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py"""
def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


"""This function below is copied from https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py, but revised
so that the score_funcs are calculated after each batch rather than at the end. Now, at the end, the scores from each
batch are averaged. The reason for this change is for memory. Previously, the y_pred and y_true were being appended to a list
after each batch which required much more space than I have available."""
def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs. 
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary. 
    desc -- a description to use for the progress bar.     
    """
    running_loss = []
    # make a dictionary of empty lists for each score_func
    score_func_batch_results = {}
    for name, score_func in score_funcs.items():
      score_func_batch_results[name] = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #Move the batch to the device we are using. 
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs) #this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Now we are just grabbing some information we would like to have
        running_loss.append(loss.detach().item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            #moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #add to predictions so far
            y_true = labels
            y_pred = y_hat
            # quick check for classification vs. regression,
            if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
                y_pred = np.argmax(y_pred, axis=1)
            #Else, we assume we are working on a regression problem
            # now going through each of the score functions and appending the score from this batch to it's list of values
            for name, score_func in score_funcs.items():
                try:
                  value = score_func(y_true, y_pred)
                  score_func_batch_results[name].append(value)
                except:
                  score_func_batch_results[name].append(float("NaN"))       

    #end training epoch
    end = time.time()
    
    # now that the epoch is over, average the batch scores for each score function and add them to "results" df
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( np.mean(score_func_batch_results[name]) )
        except:
            results[prefix + " " + name].append(float("NaN"))
    results[prefix + " loss"].append( np.mean(running_loss) )
    
    return end-start #time spent on epoch
  
  
"""This is direct copy from https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py so that it references the new version of 'run_epoch'
"""  
def train_network(model, loss_func, train_loader, val_loader=None, test_loader=None,score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, 
                         lr_schedule=None, optimizer=None, disable_tqdm=False
                        ):
    """Train simple neural networks
    
    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs. 
    val_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training
    lr_schedule -- the learning rate schedule used to alter \eta as the model trains. If this is not None than the user must also provide the optimizer to use. 
    optimizer -- the method used to alter the gradients for learning. 
    
    """
    if score_funcs == None:
        score_funcs = {}#Empty set 
    
    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if val_loader is not None:
            to_track.append("val " + eval_score )
        if test_loader is not None:
            to_track.append("test "+ eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []

        
    if optimizer == None:
        #The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()#Put our model in training mode

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")
        
        results["epoch"].append( epoch )
        results["total time"].append( total_train_time )
        
      
        if val_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")
                
        #In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()
                
        if test_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
        
        
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results' : results
                }, checkpoint_file)
    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)
  
def get_auc(masks,predictions):
  sigmoid_layer = nn.Sigmoid()
  if predictions.shape != masks.shape:
    raise ValueError(f"Predictions and Masks have different shapes. Predictions have shape {predictions.shape} and masks have shape of {masks.shape}")
  predictions = sigmoid_layer(torch.tensor(predictions))
  predictions = predictions.detach().numpy()
  class_accuracies = []
  for c in range(predictions.shape[1]): # dimension 0 will be batch so class dimension will be 1
    flattened_pred_c = predictions[:,c,:,:].flatten()
    flattened_mask_c = masks[:,c,:,:].flatten()
    fpr, tpr, thresholds = roc_curve(flattened_mask_c,flattened_pred_c)
    class_accuracies.append(auc(fpr,tpr))
  return np.nanmean(class_accuracies)

def get_accuracy(masks,predictions):
  sigmoid_layer = nn.Sigmoid()
  if predictions.shape != masks.shape:
    raise ValueError(f"Predictions and Masks have different shapes. Predictions have shape {predictions.shape} and masks have shape of {masks.shape}")
  predictions = sigmoid_layer(torch.tensor(predictions))
  predictions = predictions.detach().numpy()
  class_accuracies = []
  for c in range(predictions.shape[1]): # dimension 0 will be batch so class dimension will be 1
    flattened_pred_c = predictions[:,c,:,:].flatten()
    flattened_mask_c = masks[:,c,:,:].flatten()
    class_accuracies.append(accuracy_score(flattened_mask_c,(flattened_pred_c>0.5).astype(int)))
  return np.nanmean(class_accuracies)
