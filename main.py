import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil
import cv2
import matplotlib.image as mpimg
import seaborn as sns 
import zipfile
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow .keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print(cv2.__version__)
print(os.getcwd())

target_dirr = os.path.join(os.getcwd(), "./TumorTekrar")

os.chdir(target_dirr)
print("yeni çalışma yerimiz: ", os.getcwd())


z = zipfile.ZipFile(".\\archive.zip")
z.extractall()

folder = "brain_tumor_dataset/yes/"
count = 1

for filename in os.listdir(folder):
    source = folder+filename
    destination = folder + "Y_" + str(count) + ".jpg"
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping...")
    else: 
        os.rename(source, destination)
        count += 1

print("All files in the yes directory got numbered. ")

folder_no = "brain_tumor_dataset/no/"
count = 1
for filename in os.listdir(folder_no):
    source = folder_no+filename
    destination = folder_no + "N_" + str(count) + ".jpg"
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping...")
    else:

        os.rename(source, destination)
        count+= 1
print("All files in the no directory got numbered.")    

"""
Exploratory Data Analysis (EDA): Model geliştirirken veri setiini 
anlamak ve sorunlari giderme amacli yapilir. Mesela burada 
veri setimiz küçük olduğu için DL ile bu sayiyi arttiracagiz. 
Ancak bunun için önce grafikteki dagilimlarina bakmak gerekir. 
"""

listyes = os.listdir("brain_tumor_dataset/yes/")
print(type(listyes))
number_files_yes = len(listyes)
print("Number of the yes files:", number_files_yes)

listno = os.listdir("brain_tumor_dataset/no/")
number_files_no = len(listno)
print("Number of the no files: ", number_files_no)

#Bu sekilde sayi olarak da görebiliriz. Ama grafik olarak görmek daha cok isimize yarar.abs

# For this purpose we create a dictionary.

data = {"tumorous" : number_files_yes,
        "non-tumorous" : number_files_no}
typex = data.keys()
values = data.values()

fig = plt.figure(figsize = (8,6))
plt.bar(typex, values, color = "blue")
plt.xlabel("Number of Brain MRI Images")
plt.ylabel("Data")
plt.title("Count of Brain Tumor Images")
plt.show()


# 155 tümörlü, 98 tümörü olmayan beyin resmi.
## Yüzdelik olarak %61,26 tümörlü, %38,7 benzeri bir oran var. 
### Bunun için de keras ve tensorflow yapılabilir. 

def timing(sec_elapsed):
    h = int(sec_elapsed / (60*60))
    m = int(sec_elapsed % (60*60)/60)
    s = sec_elapsed % 60
    return f"{h} : {m} : {s}"


def augmented_data(file_dir, n_generated_samples, save_to_dir):
    os.makedirs(save_to_dir, exit_ok = True)
    data_gen = ImageDataGenerator(rotation_range = 10,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1, 
                                  shear_range = 0.1,
                                  brightness_range = (0.3,1.0),
                                  horizontal_flip = True,
                                  fill_mode = "nearest")


    for filename in os.listdir(file_dir):
        image = cv2.imread(file_dir + "/" + filename)
        image = image.reshape((1,)+ iamge.shape)
        save_prefix = "aug_" + filename[:-4]
        i = 0
        for batch in data_gen.flow(x = image, batch_size = 1, save_to_dir = save_to_dir,
                                    save_prefix = save_prefix, save_format = "jpg"):
            i += 1
            if i>n_generated_samples:
                break