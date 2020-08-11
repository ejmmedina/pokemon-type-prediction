![heading.png](heading.png)

# Executive Summary

Pokémon is a video game franchise that has existed for more than two decades now. Today, there are more than 800 Pokemon and 18 different Pokémon types, spanning 7 generations. With each new generation, dozens of Pokemons are introduced, with varying types, making it a problem for those without Pokemon knowledge to follow the franchise. As such, a machine learning approach is taken to identify a Pokemon's type, given its image.

The data consists of 120 by 120 px images of the 809 Pokemon currently introduced. Each image is processed in order to extract features containing its color and size information. In doing so, we minimize the features from 43200 features to 21. Aside from the images, there is also data on the names and their respective types, which we use as targets.

The analysis is done twice: one for the original data, and another undersampled to balance the data. Logistic Regression with L1 regularization is used for the original data while Linear SVM with L1 regularization is used for the balanced data due to having the highest accuracy. From the resulting models, the predictability of different Pokemon types and generations are observed.

From this analysis, we observe that different combination of features lead to different accuracies, and more features does not necessarily lead to higher accuracy since some features introduce noise.

The overall accuracy of the models are not worth mentioning as they range from 10% to 40% but breaking these down to the respective classes will yield accuracies ranging from 0% to 80%. This indicates that the features designed are predictive of some classes but not all.

The analysis done in this project is only applied to Pokemon data but it can be applied to other datasets. The ideas and learnings obtained from this project can be useful for analysis and other machine learning problems to be encountered in the future.


# Table of Contents
1. <a href='#introduction'>Introduction</a>
2. <a href='#data'>Data Description</a>
3. <a href='#eda'>EDA and Feature Extraction</a><br>
    3a. <a href='#color'>Color features</a><br>
    3b. <a href='#size'>Size features</a>
4. <a href='#imbalanced'>Classification using imbalanced dataset</a><br>
    4a. <a href='#imb-main'>Main type as target</a><br>
    4b. <a href='#imb-sec'>Secondary type as target</a><br>
    4c. <a href='#imb-sum'>Summary (Imbalanced)</a>
5. <a href='#balanced'>Data balancing through undersampling</a><br>
    5a. <a href='#bal-main'>Main type as target</a><br>
    5b. <a href='#bal-sec'>Secondary type as target</a><br>
    5c. <a href='#bal-sum'>Summary (Balanced)</a>
6. <a href='#conclusion'>Conclusion and insights</a><br>
<a href='#ref'>References</a><br>
<a href='#ack'>Acknowledgements</a><br>
<a href='#appendix'>Appendix</a><br>



<a id="introduction"></a>
# Introduction

**Pokémon**, short for Pocket Monsters, is a video game franchise created in 1995 but is still releasing versions up until today, with an upcoming version set to release November this year. With Pokémon expanding to other media like movies and mobile apps, it has reached the global market, even capturing those who are not the main followers of the franchise. For them, it is difficult to catch up to the system of Pokémon, particularly their type system.

The type of some Pokémon can easily be identified based on their color schemes, red for fire, blue for water, green for grass, and the like. However, this is not true for all and so, other information like the presence of wings, size of the Pokemon, and the like, must also be extracted.

Most image recognition processes are done through deep learning or by using the pixels of the images as features. However, processing this type of data can be computationally heavy due to the sheer number of features ($\text{Image Dimension} \times 3$). By reducing the number of features, we are also reducing the accuracy and the complexity at the same time, offering a cheaper way to classify images based on information like color theme and size information.

<a id="data"></a>
# Data Description
The dataset consists of images of 809 Pokemons (all the currently released Pokemons), downloaded from Kaggle. Each image depicts the Pokemon with a white background. There are two file types for the images, JPG and PNG, with the former for Pokemons in Gen VII while PNG is the format for the rest.

Aside from the images, a sheet containing the Pokemon names and their respective main and secondary types are also included in the dataset, ordered according to their National Pokedex number. To check the generation of the Pokemon, their Pokedex numbers are cross-referenced with Pokemon's wikipedia - bulbapedia.

There are 18 Pokemon types, any combination of which can give the Primary and Secondary type of a Pokemon. An important thing to note is that the 18 types were only completed in Gen VI. In Gen I, there are only 15 types, with Dark and Steel introduced in Gen II and Fairy introduced in Gen VI.

The csv file looks like this:


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import mglearn #library provided by amueller
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
```

    C:\Users\Justin\Anaconda3\lib\site-packages\sklearn\externals\six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    C:\Users\Justin\Anaconda3\lib\site-packages\sklearn\externals\joblib\__init__.py:15: FutureWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.
      warnings.warn(msg, category=FutureWarning)



```python
df = pd.read_csv('pokemon.csv')
# df.drop('Reason for absence', axis=1, inplace=True)
df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type1</th>
      <th>Type2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
    </tr>
    <tr>
      <th>2</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
    </tr>
    <tr>
      <th>3</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charizard</td>
      <td>Fire</td>
      <td>Flying</td>
    </tr>
    <tr>
      <th>6</th>
      <td>squirtle</td>
      <td>Water</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>wartortle</td>
      <td>Water</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>blastoise</td>
      <td>Water</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



<a id="eda"></a>
# EDA and Feature extraction

The pokemons are actually ordered by their (national) pokedex number, and so, we can get their Pokedex Number from the index. Using these Pokedex Numbers, we can infer their Generation:


```python
df['Pokedex Number'] = df.index + 1
df.set_index('Pokedex Number', inplace=True)
```


```python
df.loc[1:151, 'Generation'] = 1
df.loc[152:251, 'Generation'] = 2
df.loc[252:386, 'Generation'] = 3
df.loc[387:493, 'Generation'] = 4
df.loc[494:649, 'Generation'] = 5
df.loc[650:721, 'Generation'] = 6
df.loc[722:809, 'Generation'] = 7
df.sample(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type1</th>
      <th>Type2</th>
      <th>Generation</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>arbok</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>658</th>
      <td>greninja</td>
      <td>Water</td>
      <td>Dark</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>217</th>
      <td>ursaring</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>bayleef</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from PIL import Image

```

<a id="color"></a>

## Color feature
Possible features:

    - RGB channels histogram statistical values: mode, mean, stdev

### EDA

For the pokemon images, background is transparent for PNG files so using `im_arr.sum(axis=2)>0` is effective in order to filter the background. For JPG files, the background is white, but the edge is not clear, so we perform image processing techniques to generate the mask for JPG files

We look at the grayscale image first to filter out the edge:

#### PNG


```python
im = Image.open(f'images/zekrom.png').convert('LA')
im_arr = np.array(im)
fig, ax = plt.subplots()
ax.hist(im_arr[im_arr.mean(axis=2)<254,0], color='gray', bins=255);
# ax[2].hist(im_arr[im_arr.mean(axis=2)<255,2], color='b')

```


![png](output_img/output_17_0.png)


Looking at the very high count for grayscale value of EXACTLY 0, we don't need to do any thresholding to remove the background.

#### JPG


```python
im = Image.open(f'images/brionne.jpg').convert('LA')
im_arr = np.array(im)[:,:,0]
fig, ax = plt.subplots()
ax.hist(im_arr.flatten(), color='gray', bins=255);
# ax[2].hist(im_arr[im_arr.mean(axis=2)<255,2], color='b')

```


![png](output_img/output_20_0.png)


The white background corresponds to the grayscale value of 255. We can check if we can completely remove the background by filtering the 255 values:



```python
fig,ax = plt.subplots(1,2, figsize=(12,8))
ax[0].imshow(im_arr, interpolation=None)
ax[0].set_title('Original')
ax[1].set_title('Background vs foreground')
ax[1].imshow(im_arr<255)
```




    <matplotlib.image.AxesImage at 0x154fed78708>




![png](output_img/output_22_1.png)


However, when the pixels transition from background to foreground, there is an interpolation of colors, making it more difficult to filter the background. We can threshold the background instead:


```python
im = Image.open(f'images/brionne.jpg').convert('LA')
im_arr = np.array(im)[:,:,0]
fig, ax = plt.subplots()
ax.hist(im_arr[im_arr!=255], color='gray', bins=255);
# ax[2].hist(im_arr[im_arr.mean(axis=2)<255,2], color='b')

```


![png](output_img/output_24_0.png)


From the histogram, an appropriate threshold would be chosen by tweaking the parameters for the most difficult image to process: brionne (which is a pokemon that has alot of white elements)


```python
fig,ax = plt.subplots(1,2, figsize=(12,8))
ax[0].imshow(im_arr, interpolation=None)
ax[0].set_title('Original')
ax[1].set_title('Background vs foreground')
ax[1].imshow(im_arr<245)
```




    <matplotlib.image.AxesImage at 0x15480155388>




![png](output_img/output_26_1.png)


which is better. But it leaves spaces in the foreground.


```python
from skimage.morphology import binary_closing, binary_opening
plt.figure()
plt.imshow(im_arr<251, cmap='gray')
plt.axis('off')
# plt.savefig('threshold.png', dpi=300, bbox_inches='tight')
plt.figure()
plt.imshow(im_arr, cmap='gray')
plt.axis('off')
# plt.savefig('orig.png', dpi=300, bbox_inches='tight')
plt.figure()
neighborhood = np.ones((5,5))
neighborhood[0, 0] = 0
neighborhood[0, -1] = 0
neighborhood[-1, 0] = 0
neighborhood[-1, -1] = 0
mask = binary_closing(binary_opening(im_arr<251), neighborhood)
plt.imshow(mask, cmap='gray')
plt.axis('off')
# plt.savefig('processed.png', dpi=300, bbox_inches='tight')
plt.figure()
im
```




![png](output_img/output_28_0.png)




![png](output_img/output_28_1.png)



![png](output_img/output_28_2.png)



![png](output_img/output_28_3.png)



    <Figure size 432x288 with 0 Axes>



```python
plt.imshow(np.ma.masked_array(im_arr[:,:], (im_arr<250)), interpolation=None)
```




    <matplotlib.image.AxesImage at 0x154fea85c88>




![png](output_img/output_29_1.png)



```python
from scipy import stats
```

For each color channel, we extract the following information:
- mean
- mode and the frequency at the mode
- standard deviation


```python
im = Image.open(f'images/pikachu.png').convert('RGBA')
im_arr = np.array(im)
fig, ax = plt.subplots()
ax.hist(im_arr[im_arr.sum(axis=2)>0,2], bins=20, color='gray', alpha=0.7)
ax.set_xlabel('Pixel value', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.axvline(im_arr[im_arr.sum(axis=2)>0, 2].mean(), color='red', ls='--', lw=3, label='Mean')
ax.axvline(stats.mode(im_arr[im_arr.sum(axis=2)>0, 2])[0][0], color='green', ls='--', lw=3, label='Mode')
ax.plot([im_arr[im_arr.sum(axis=2)>0, 2].mean() - im_arr[im_arr.sum(axis=2)>0, 2].std(),
          im_arr[im_arr.sum(axis=2)>0, 2].mean() + im_arr[im_arr.sum(axis=2)>0, 2].std()], [200, 200], 'b--', lw=3, label='Standard Deviation')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
```




    <matplotlib.legend.Legend at 0x154fbc337c8>




![png](output_img/output_32_1.png)



```python
from IPython.display import clear_output
def plot_pokemon(pokemon):
    im = Image.open(f'images/{pokemon}.PNG').convert('RGBA')
    im_arr = np.array(im)
    fig, ax = plt.subplots(1, 3, figsize=(16,4))
    ax[0].hist(im_arr[im_arr.sum(axis=2)>0,0], color='r',bins=15)
    ax[0].set_xlabel('Red Pixel Values')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_ylabel('Count')
    ax[0].axis('off')
    ax[1].hist(im_arr[im_arr.sum(axis=2)>0,1], color='g',bins=15)
    ax[1].set_xlabel('Green Pixel Values')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_ylabel('Count')
    ax[1].axis('off')
    ax[2].hist(im_arr[im_arr.sum(axis=2)>0,2], color='b',bins=15)
    ax[2].set_xlabel('Blue Pixel Values')
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].set_ylabel('Count')
    ax[2].axis('off')
    ax[0].set_title(pokemon, loc='left', size=16)
    return im_arr, ax
```


```python
plot_pokemon('bulbasaur');
```


![png](output_img/output_34_0.png)



```python
plot_pokemon('charmander');
```


![png](output_img/output_35_0.png)



```python
plot_pokemon('squirtle');
```


![png](output_img/output_36_0.png)



```python
plot_pokemon('pikachu');
```


![png](output_img/output_37_0.png)


### Feature extraction

For this image, we will extract different features:


```python
im = Image.open(f'images/pikachu.PNG').convert('RGBA')
im_arr = np.array(im)
im_arr_pokemon = im_arr[im_arr.sum(axis=2)>0]
plt.imshow(im_arr)
```




    <matplotlib.image.AxesImage at 0x154fed41a48>




![png](output_img/output_39_1.png)


#### Mean <br>(red, green, blue)


```python
def get_mean(im_arr_pokemon):
    mean_r = im_arr_pokemon[:, 0].mean()
    mean_g = im_arr_pokemon[:, 1].mean()
    mean_b = im_arr_pokemon[:, 2].mean()
    return (mean_r, mean_g, mean_b)
get_mean(im_arr_pokemon)
```




    (220.36327458783398, 204.3939738487777, 116.59351904491189)



#### Mode and Mode Frequency <br>(mode_red, mode_green, mode_blue, frequency at mode_red, frequency at mode_green, frequency at mode_blue)


```python
from scipy import stats
```


```python
def get_mode_and_freq(im_arr_pokemon):
    mode_r = stats.mode(im_arr_pokemon[:, 0])[0][0]
    mode_g = stats.mode(im_arr_pokemon[:, 1])[0][0]
    mode_b = stats.mode(im_arr_pokemon[:, 2])[0][0]
    n_mode_r = (im_arr_pokemon[:, 0]==mode_r).sum() / len(im_arr_pokemon[:, 0])
    n_mode_g = (im_arr_pokemon[:, 1]==mode_g).sum() / len(im_arr_pokemon[:, 0])
    n_mode_b = (im_arr_pokemon[:, 2]==mode_b).sum() / len(im_arr_pokemon[:, 0])
    return (mode_r, mode_g, mode_b, n_mode_r, n_mode_g, n_mode_b)
get_mode_and_freq(im_arr_pokemon)
```




    (255, 238, 119, 0.46219442865264354, 0.2984650369528141, 0.38828880045480385)



#### Standard deviation<br>(red, green, blue)


```python
def get_std(im_arr_pokemon):
    std_r = im_arr_pokemon[:, 0].std()
    std_g = im_arr_pokemon[:, 1].std()
    std_b = im_arr_pokemon[:, 2].std()
    return (std_r, std_g, std_b)
get_std(im_arr_pokemon)
```




    (45.50596652018251, 45.91241616382711, 37.047856871336265)



Now we extract the color features for all the Pokemon images.


```python

import re
import os

files_ = next(os.walk('images'))[2]
png_or_jpg = dict(re.findall('(^.*)\.(.*$)', '\n'.join(files_), re.M))
df_feat_color = pd.DataFrame(columns=['mean_r', 'mean_g', 'mean_b', 'mode_r', 'mode_g', 'mode_b',
              'mode_freq_r', 'mode_freq_g', 'mode_freq_b', 'std_r', 'std_g',
               'std_b', 'png'], index=df.index)


neighborhood = np.ones((5,5))
neighborhood[0, 0] = 0
neighborhood[0, -1] = 0
neighborhood[-1, 0] = 0
neighborhood[-1, -1] = 0

for i in df.index:
    pokemon = df.loc[i].Name
    im = Image.open(f'images/{pokemon}.{png_or_jpg[pokemon]}').convert('RGBA')
    im_arr = np.array(im)
    if png_or_jpg[pokemon]=='png':
        mask = im_arr.sum(axis=2)>0
    elif png_or_jpg[pokemon]=='jpg':
        imgray = Image.open(f'images/{pokemon}.jpg').convert('LA')
        im_gray_arr = np.array(imgray)[:,:,0]
        mask = binary_closing(binary_opening(im_gray_arr<251), neighborhood)
    im_arr_pokemon = im_arr[mask]
    means = get_mean(im_arr_pokemon)
    modes = get_mode_and_freq(im_arr_pokemon)
    df_feat_color.loc[i] = [*means,
                            *modes,
                            *get_std(im_arr_pokemon),
                            png_or_jpg[pokemon]=='png'
                            ]

```

Checking the filetypes,


```python
dummy = df_feat_color.copy()
dummy['Generation'] = df['Generation']

dummy.groupby('Generation')['png'].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>151</td>
      <td>1</td>
      <td>True</td>
      <td>151</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>100</td>
      <td>1</td>
      <td>True</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>135</td>
      <td>1</td>
      <td>True</td>
      <td>135</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>107</td>
      <td>1</td>
      <td>True</td>
      <td>107</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>156</td>
      <td>1</td>
      <td>True</td>
      <td>156</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>72</td>
      <td>1</td>
      <td>True</td>
      <td>72</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>88</td>
      <td>1</td>
      <td>False</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>



Upon closer inspection, the images with PNG filetypes are those from Gen I to Gen VI pokemons while Gen VII pokemons have JPG filetypes.


```python
df_feat_color.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_r</th>
      <th>mean_g</th>
      <th>mean_b</th>
      <th>mode_r</th>
      <th>mode_g</th>
      <th>mode_b</th>
      <th>mode_freq_r</th>
      <th>mode_freq_g</th>
      <th>mode_freq_b</th>
      <th>std_r</th>
      <th>std_g</th>
      <th>std_b</th>
      <th>png</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>141.361</td>
      <td>192.116</td>
      <td>161.39</td>
      <td>153</td>
      <td>221</td>
      <td>187</td>
      <td>0.295185</td>
      <td>0.249128</td>
      <td>0.179344</td>
      <td>39.0903</td>
      <td>47.1262</td>
      <td>44.0586</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>144.255</td>
      <td>155.1</td>
      <td>161.621</td>
      <td>136</td>
      <td>119</td>
      <td>136</td>
      <td>0.180596</td>
      <td>0.185879</td>
      <td>0.15658</td>
      <td>68.465</td>
      <td>56.7961</td>
      <td>59.4548</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>145.169</td>
      <td>175.54</td>
      <td>169.631</td>
      <td>153</td>
      <td>221</td>
      <td>238</td>
      <td>0.239791</td>
      <td>0.175797</td>
      <td>0.133466</td>
      <td>50.8307</td>
      <td>57.3943</td>
      <td>65.4917</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>213.607</td>
      <td>176.345</td>
      <td>127.693</td>
      <td>255</td>
      <td>187</td>
      <td>136</td>
      <td>0.37781</td>
      <td>0.188542</td>
      <td>0.195069</td>
      <td>49.409</td>
      <td>45.0129</td>
      <td>46.5492</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>218.464</td>
      <td>151.415</td>
      <td>123.425</td>
      <td>255</td>
      <td>153</td>
      <td>136</td>
      <td>0.410642</td>
      <td>0.161943</td>
      <td>0.149219</td>
      <td>43.2752</td>
      <td>50.9202</td>
      <td>58.7825</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



After extracting the features, we look at the correlation matrix of these features


```python
df_feat_color.columns
```




    Index(['mean_r', 'mean_g', 'mean_b', 'mode_r', 'mode_g', 'mode_b',
           'mode_freq_r', 'mode_freq_g', 'mode_freq_b', 'std_r', 'std_g', 'std_b',
           'png'],
          dtype='object')




```python
# Compute the correlation matrix
import seaborn as sns
df_feat_color = df_feat_color.astype(float)
corr = df_feat_color[df_feat_color.columns[:]].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 7))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
ax.axes.set_title("Correlation Matrix of Color Features", fontsize=18, y=1.01);
```


![png](output_img/output_55_0.png)


Note that there are heavily correlated features like mean and mode of the respective channel, which is expected since there are many instances where the mean is very close to the mode. However, the fact that it is not always the same indicates that we can still use these separate features, since machine learning techniques have ways to deal with correlation (regularization).

What is unexpected here is the correlation of the RGB modes and the png to the modes. Note that the mode is very susceptible to improper foreground and background isolation. In the interest of time, we will employ the basic thresholding and morphological operations but we can make several recommendations to improve this result.

<a id="size"></a>
## Size feature

Aside from the colors, we can also extract the sizes through pixels since the images share the same dimensions (120 by 120). Though the sizes proportions among the pokemons are not exactly scaled appropriately, there is still a difference in the sizes, which we can use as features

### EDA
Comparing two Pokemons with significantly different size and shape,


```python
im_large = Image.open('images/guzzlord.jpg').convert('RGBA')
im_arr_large = np.array(im_large)

imgray = Image.open(f'images/guzzlord.jpg').convert('LA')
im_gray_arr = np.array(imgray)[:,:,0]
mask_large = binary_closing(binary_opening(im_gray_arr<251), neighborhood)

im_small = Image.open('images/wingull.png').convert('RGBA')
im_arr_small = np.array(im_small)
mask_small = im_arr_small.sum(axis=2)>0
```


```python
fig, ax = plt.subplots(1,2,figsize=(12,7))
ax[0].imshow(im_arr_large)
ax[1].imshow(im_arr_small)
```




    <matplotlib.image.AxesImage at 0x15483e9b408>




![png](output_img/output_60_1.png)


However, as expected, while guzzlord (left) is larger than wingull (right), it is not to scale since guzzlord (according to wiki) is 5.5m tall while anorith is 0.61m. Nonetheless, we will use the pixels as proxies.

Foreground


```python
plt.imshow(mask_small, cmap='Pastel1')
plt.axis('off')
```




    (-0.5, 119.5, 119.5, -0.5)




![png](output_img/output_63_1.png)


Background


```python
plt.imshow(mask_small, cmap='Oranges')
plt.axis('off')
# plt.savefig('foreground.png', dpi=300, bbox_inches='tight')
```


![png](output_img/output_65_0.png)



```python
from matplotlib import cm
```

Nonetheless, there are still observable differences which is why it may be necessary to extract information like height and width information.


```python
fig, ax = plt.subplots()
ax.bar(range(120),(mask_small).sum(axis=0), color=(0.9998769703960015, 0.4589388696655133, 0.4180007689350249, 1.0)) #height
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Height (in px)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Height distribution', loc='left', size=16);
# ax.axis('off');
```


![png](output_img/output_68_0.png)



```python
fig, ax = plt.subplots()
ax.bar(range(120),(mask_small).sum(axis=1), color=(0.9998769703960015, 0.4589388696655133, 0.4180007689350249, 1.0)) #height
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Height (in px)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Height distribution', loc='left', size=16);
# ax.axis('off');
```




    Text(0.0, 1.0, 'Height distribution')




![png](output_img/output_69_1.png)



```python
fig, ax = plt.subplots()
ax.hist((im_arr_small>0).sum(axis=1)[(im_arr_small>0).sum(axis=1)>0], color=(0.9998769703960015, 0.4589388696655133, 0.4180007689350249, 1.0)) #height
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Width (in px)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Width distribution', loc='left', size=16);
# ax.axis('off')
```


![png](output_img/output_70_0.png)


### Feature extraction

For the two Pokemons (Guzzlord and Wingull), we extract the following features:

#### Size

Guzzlord


```python
mask_large.sum()
```




    7413



Wingull


```python
mask_small.sum()
```




    783



#### Height


```python
def height_info(mask):
    height = mask.sum(axis=0)[mask.sum(axis=0)>0]
    return (np.mean(height), stats.mode(height)[0][0], np.std(height))
```

Wingull (mean, mode, standard deviation)


```python
height_info(mask_small)
```




    (6.75, 6, 3.142794166408897)



Guzzlord (mean, mode, standard deviation)


```python
height_info(mask_large)
```




    (61.775, 74, 27.09257662780219)



#### Width


```python
def width_info(mask):
    width = mask.sum(axis=1)[mask.sum(axis=1)>0]
    return (np.mean(width), stats.mode(width)[0][0], np.std(width))
```

Wingull (mean, mode, standard deviation)


```python
width_info(mask_small)
```




    (46.05882352941177, 18, 37.329172613757216)



Guzzlord (mean, mode, standard deviation)


```python
width_info(mask_large)
```




    (80.57608695652173, 91, 25.38624828680668)



Now we extract the size features from the Pokemon images


```python
files_ = next(os.walk('images'))[2]
png_or_jpg = dict(re.findall('(^.*)\.(.*$)', '\n'.join(files_), re.M))

df_feat_size = pd.DataFrame(columns=['size', 'mean_width', 'mode_width', 'std_width',
                                    'mean_length', 'mode_length', 'std_length'], index=df.index)


neighborhood = np.ones((5,5))
neighborhood[0, 0] = 0
neighborhood[0, -1] = 0
neighborhood[-1, 0] = 0
neighborhood[-1, -1] = 0

for i in df.index:
    pokemon = df.loc[i].Name
    im = Image.open(f'images/{pokemon}.{png_or_jpg[pokemon]}').convert('RGBA')
    im_arr = np.array(im)
    if png_or_jpg[pokemon]=='png':
        mask = im_arr.sum(axis=2)>0
    elif png_or_jpg[pokemon]=='jpg':
        imgray = Image.open(f'images/{pokemon}.jpg').convert('LA')
        im_gray_arr = np.array(imgray)[:,:,0]
        mask = binary_closing(binary_opening(im_gray_arr<251), neighborhood)
    size = mask.sum()
    df_feat_size.loc[i] = [size,
                           *width_info(mask),
                           *height_info(mask)]
```


```python
df_feat_size.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>mean_width</th>
      <th>mode_width</th>
      <th>std_width</th>
      <th>mean_length</th>
      <th>mode_length</th>
      <th>std_length</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1433</td>
      <td>31.1522</td>
      <td>38</td>
      <td>10.8907</td>
      <td>34.119</td>
      <td>38</td>
      <td>10.2544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2082</td>
      <td>33.5806</td>
      <td>39</td>
      <td>12.9758</td>
      <td>28.1351</td>
      <td>3</td>
      <td>23.234</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4016</td>
      <td>57.3714</td>
      <td>75</td>
      <td>20.7269</td>
      <td>42.2737</td>
      <td>63</td>
      <td>20.9226</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1379</td>
      <td>24.625</td>
      <td>26</td>
      <td>6.99888</td>
      <td>33.6341</td>
      <td>51</td>
      <td>19.2389</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1729</td>
      <td>27.0156</td>
      <td>27</td>
      <td>10.2416</td>
      <td>37.587</td>
      <td>57</td>
      <td>18.169</td>
    </tr>
  </tbody>
</table>
</div>



## Combine!


```python
df_feat = df_feat_color.join(df_feat_size)
df_feat['Generation'] = df['Generation']
```


```python
df_feat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_r</th>
      <th>mean_g</th>
      <th>mean_b</th>
      <th>mode_r</th>
      <th>mode_g</th>
      <th>mode_b</th>
      <th>mode_freq_r</th>
      <th>mode_freq_g</th>
      <th>mode_freq_b</th>
      <th>std_r</th>
      <th>...</th>
      <th>std_b</th>
      <th>png</th>
      <th>size</th>
      <th>mean_width</th>
      <th>mode_width</th>
      <th>std_width</th>
      <th>mean_length</th>
      <th>mode_length</th>
      <th>std_length</th>
      <th>Generation</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>141.361479</td>
      <td>192.115841</td>
      <td>161.390091</td>
      <td>153.0</td>
      <td>221.0</td>
      <td>187.0</td>
      <td>0.295185</td>
      <td>0.249128</td>
      <td>0.179344</td>
      <td>39.090272</td>
      <td>...</td>
      <td>44.058603</td>
      <td>1.0</td>
      <td>1433</td>
      <td>31.1522</td>
      <td>38</td>
      <td>10.8907</td>
      <td>34.119</td>
      <td>38</td>
      <td>10.2544</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>144.254563</td>
      <td>155.099904</td>
      <td>161.620557</td>
      <td>136.0</td>
      <td>119.0</td>
      <td>136.0</td>
      <td>0.180596</td>
      <td>0.185879</td>
      <td>0.156580</td>
      <td>68.464976</td>
      <td>...</td>
      <td>59.454752</td>
      <td>1.0</td>
      <td>2082</td>
      <td>33.5806</td>
      <td>39</td>
      <td>12.9758</td>
      <td>28.1351</td>
      <td>3</td>
      <td>23.234</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>145.169074</td>
      <td>175.539841</td>
      <td>169.630976</td>
      <td>153.0</td>
      <td>221.0</td>
      <td>238.0</td>
      <td>0.239791</td>
      <td>0.175797</td>
      <td>0.133466</td>
      <td>50.830729</td>
      <td>...</td>
      <td>65.491676</td>
      <td>1.0</td>
      <td>4016</td>
      <td>57.3714</td>
      <td>75</td>
      <td>20.7269</td>
      <td>42.2737</td>
      <td>63</td>
      <td>20.9226</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>213.606962</td>
      <td>176.345178</td>
      <td>127.692531</td>
      <td>255.0</td>
      <td>187.0</td>
      <td>136.0</td>
      <td>0.377810</td>
      <td>0.188542</td>
      <td>0.195069</td>
      <td>49.409046</td>
      <td>...</td>
      <td>46.549167</td>
      <td>1.0</td>
      <td>1379</td>
      <td>24.625</td>
      <td>26</td>
      <td>6.99888</td>
      <td>33.6341</td>
      <td>51</td>
      <td>19.2389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>218.464430</td>
      <td>151.414691</td>
      <td>123.424523</td>
      <td>255.0</td>
      <td>153.0</td>
      <td>136.0</td>
      <td>0.410642</td>
      <td>0.161943</td>
      <td>0.149219</td>
      <td>43.275183</td>
      <td>...</td>
      <td>58.782531</td>
      <td>1.0</td>
      <td>1729</td>
      <td>27.0156</td>
      <td>27</td>
      <td>10.2416</td>
      <td>37.587</td>
      <td>57</td>
      <td>18.169</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



<a id="imbalanced"></a>

# Classification using imbalanced dataset
Targets:
1. Primary type only
2. Secondary type only

It is possible to predict the combination of both but that would lead to more classes, making the problem more difficult.

<a id="imb-main"></a>

### Primary Type only


```python
df_target_A = df['Type1']
```


```python
df_target_A.value_counts()[::-1].plot.barh()
plt.xlabel('Count')
```




    Text(0.5, 0, 'Count')




![png](output_img/output_98_1.png)


Looking at the distribution of the target classes, it is immediately obvious that the dataset is imbalanced. Particularly with Flying Pokemons, this type is usually a Secondary Type rather than a Primary Type, causing it to have a low count. Nonetheless, we will continue with this data and observe the results.


For this analysis, we use three combinations of features:
1. All the features
2. Only the color features
3. Only the size features

in order to predict the primary type of the Pokemon

#### Combined features



```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

import importlib

from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

```


```python
print(f'1.25 x PCC = {125*((df_target_A.value_counts() / df_target_A.count())**2).sum():.2f}%')
```

    1.25 x PCC = 9.83%



```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat.astype(float)
y = df_target_A

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('KNN', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                         weights='uniform'), {'n_neighbors': range(1, 31)})

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Logistic Regression (L2)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Linear SVM (L2)', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('NonLinear SVM (RBF)', SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05]), 'gamma': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('NonLinear SVM (Poly)', SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05]), 'gamma': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Decision Tree', DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best'), {'max_depth': range(1, 11), 'criterion': ['gini', 'entropy']})

    Training ('Random Forest', RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='sqrt',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False), {'max_depth': range(1, 4), 'n_estimators': range(10, 101, 10), 'criterion': ['gini', 'entropy']})

    Training ('GBM', GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='deprecated',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False), {'max_depth': range(1, 4), 'n_estimators': range(10, 101, 10), 'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_all = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_all.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_all
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>0.206897</td>
      <td>{'n_neighbors': 23}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression (L1)</td>
      <td>0.261084</td>
      <td>{'C': 3.593813663804626}</td>
      <td>png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression (L2)</td>
      <td>0.231527</td>
      <td>{'C': 3.593813663804626}</td>
      <td>png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Linear SVM (L1)</td>
      <td>0.290640</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>mean_b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Linear SVM (L2)</td>
      <td>0.280788</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>png</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NonLinear SVM (RBF)</td>
      <td>0.275862</td>
      <td>{'C': 46.41588833612782, 'gamma': 0.0016681005...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NonLinear SVM (Poly)</td>
      <td>0.211823</td>
      <td>{'C': 0.0001291549665014884, 'gamma': 3.593813...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Decision Tree</td>
      <td>0.197044</td>
      <td>{'criterion': 'gini', 'max_depth': 5}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Random Forest</td>
      <td>0.226601</td>
      <td>{'criterion': 'gini', 'max_depth': 3, 'n_estim...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GBM</td>
      <td>0.206897</td>
      <td>{'learning_rate': 0.2, 'max_depth': 1, 'n_esti...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



From the different models, Logistic Regression with L1 regularization is the best. Looking at the weights:


```python
coefs = models['Logistic Regression (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.6919057517439353, 0.6919057517439353)




![png](output_img/output_110_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_all_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'
for i in df_target_A.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_all_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```

Looking at the accuracies for the different classifications:


```python
df_acc_prim_all_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.533333</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.391304</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.384615</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_all_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_all_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_prim_all_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_prim_all_gen.set_index('Generation', inplace=True)
df_acc_prim_all_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.368421</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.318182</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.340909</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.238095</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.142857</td>
    </tr>
  </tbody>
</table>
</div>



#### Color features only
Using a subset of the features, what would the accuracies be?


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type1</th>
      <th>Type2</th>
      <th>Generation</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df_feat_color
X['Generation'] = df['Generation'].astype(float)
y = df_target_A

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```

To make the analysis more comparable, we will use the same model for the different feature subsets.


```python
models = {}
for est in estimators[1:2]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_color = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_color.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_color
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.256158</td>
      <td>{'C': 599.4842503189421}</td>
      <td>png</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_color_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'
for i in df_target_A.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_color_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_color_types['Top Predictor Variable']['Water']
```




    {'mean_r': 1.1604741401267666,
     'mean_g': 1.7687725597116122,
     'mean_b': 1.3968118486004508,
     'mode_r': 0.6018755526150712,
     'mode_g': 1.0195660683900658,
     'mode_b': 1.0741450690455625,
     'mode_freq_r': 0.4594121067813701,
     'mode_freq_g': 0.4944337652792616,
     'mode_freq_b': 1.2295899728734687,
     'std_r': 0.9625086617491027,
     'std_g': 0.9256276412039706,
     'std_b': 0.3721836533037734,
     'png': 3.813710067315665,
     'Generation': 1.2157918434370218}




```python
coefs = models['Logistic Regression (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_color.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-2.968341979445716, 2.968341979445716)




![png](output_img/output_126_1.png)


Interestingly, the top feature for the logistic regression is the file type of the image processed (PNG or JPG). As observed earlier, the PNG is for pokemons from Gen I to Gen VI while the filetype of the most recent Gen VII are all JPG. While this is not a color feature, it is interesting to see how it is a more significant feature than the colors (which could be attributed to the difference in style from the first generations to the last ones). Note that Gen VII is where Alolan pokemons are introduced, signifying a change in style which was captured by this model.


```python
df_acc_prim_color_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.478261</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.307692</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.607143</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_color_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_color_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_prim_color_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_prim_color_gen.set_index('Generation', inplace=True)
df_acc_prim_color_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.342105</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.409091</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.212121</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.190476</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.238095</td>
    </tr>
  </tbody>
</table>
</div>



#### Size features only


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat_size
X['Generation'] = df['Generation'].astype(float)
y = df_target_A

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[1:2]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_size = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_size.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_size
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.152709</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>mean_length</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Logistic Regression (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_size.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.03220882422820588, 0.03220882422820588)




![png](output_img/output_137_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_size_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'
for i in df_target_A.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_size_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_size_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.615385</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_size_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_size_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_prim_size_gen['Top Predictor Variable'][1]
```




    {'size': 0.04273329086995648,
     'mean_width': 0.0570165363069099,
     'mode_width': 0.03463092006413958,
     'std_width': 0.015373512513715583,
     'mean_length': 0.08742663603608632,
     'mode_length': 0.06474750125990256,
     'std_length': 0.04224160384381795,
     'Generation': 0.03413330599241739}




```python
df_acc_prim_size_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_prim_size_gen.set_index('Generation', inplace=True)
df_acc_prim_size_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.263158</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.047619</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.047619</td>
    </tr>
  </tbody>
</table>
</div>



<a id="imb-sec"></a>

### Secondary Type only

#### Combined features


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

import importlib

from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

```


```python
df_target_B = df['Type2']
df_target_B[df_target_B.isnull()] = df['Type1'][df_target_B.isnull()]
```


```python
df_target_B.value_counts()[::-1].plot.barh()
plt.xlabel('Count')
```




    Text(0.5, 0, 'Count')




![png](output_img/output_147_1.png)



```python
print(f'1.25 x PCC = {125*((df_target_B.value_counts() / df_target_B.count())**2).sum():.2f}%')
```

    1.25 x PCC = 8.38%



```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat.astype(float)
y = df_target_B

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[1:2]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_all = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_all.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_all
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.236453</td>
      <td>{'C': 46.41588833612782}</td>
      <td>png</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Logistic Regression (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.8798309186889557, 0.8798309186889557)




![png](output_img/output_154_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_all_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'
for i in df_target_B.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_all_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_all_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.315789</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_all_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_all_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_sec_all_gen['Top Predictor Variable'][1]
```




    {'mean_r': 1.1309017255658198,
     'mean_g': 1.01876380808323,
     'mean_b': 0.7638530126264487,
     'mode_r': 0.8277410223760405,
     'mode_g': 0.6626352210532551,
     'mode_b': 0.7293648544757625,
     'mode_freq_r': 0.7539064357299494,
     'mode_freq_g': 0.453068676027666,
     'mode_freq_b': 0.4589230104563115,
     'std_r': 0.5246302674007338,
     'std_g': 0.5394443980717021,
     'std_b': 0.5641206905991735,
     'png': 1.5723762932090262,
     'size': 1.2349990173624372,
     'mean_width': 1.0738353984688074,
     'mode_width': 0.3798777619263852,
     'std_width': 0.4845224650436984,
     'mean_length': 0.5964029257831089,
     'mode_length': 0.4561151254276408,
     'std_length': 0.618493119668619,
     'Generation': 0.6185002819148505}




```python
df_acc_sec_all_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_sec_all_gen.set_index('Generation', inplace=True)
df_acc_sec_all_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.289474</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.121212</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.095238</td>
    </tr>
  </tbody>
</table>
</div>



#### Color features only


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type1</th>
      <th>Type2</th>
      <th>Generation</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>Fire</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>Fire</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df_feat_color
X['Generation'] = df['Generation'].astype(float)
y = df_target_B

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[1:2]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_color = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_color.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_color
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.221675</td>
      <td>{'C': 46.41588833612782}</td>
      <td>png</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Logistic Regression (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_color.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.8885043375912951, 0.8885043375912951)




![png](output_img/output_167_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_color_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'
for i in df_target_B.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_color_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_color_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.263158</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.380952</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_color_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_color_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_sec_color_gen['Top Predictor Variable'][1]
```




    {'mean_r': 1.0489141643160076,
     'mean_g': 0.9655245277154683,
     'mean_b': 0.7871276807751699,
     'mode_r': 0.8407083351620557,
     'mode_g': 0.6439534706367539,
     'mode_b': 0.7370186605874696,
     'mode_freq_r': 0.7245742359421679,
     'mode_freq_g': 0.42293578005646265,
     'mode_freq_b': 0.41548935144391475,
     'std_r': 0.4687692280370862,
     'std_g': 0.5295675721824948,
     'std_b': 0.549991652367419,
     'png': 1.436684834719361,
     'Generation': 0.6309670188545896}




```python
df_acc_sec_color_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_sec_color_gen.set_index('Generation', inplace=True)
df_acc_sec_color_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.263158</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.121212</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.295455</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.047619</td>
    </tr>
  </tbody>
</table>
</div>



#### Size features only


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat_size
X['Generation'] = df['Generation'].astype(float)
y = df_target_B

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[1:2]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_size = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_size.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_size
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.118227</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>Generation</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Logistic Regression (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_size.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.045855892880581305, 0.045855892880581305)




![png](output_img/output_179_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_size_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'
for i in df_target_B.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_size_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_size_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.368421</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.550000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_size_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Logistic Regression (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_size_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_sec_size_gen['Top Predictor Variable'][1]
```




    {'size': 0.04152374006783139,
     'mean_width': 0.08251728271158644,
     'mode_width': 0.037621426146925395,
     'std_width': 0.12446818133600407,
     'mean_length': 0.08135828626445102,
     'mode_length': 0.0783503106901134,
     'std_length': 0.08498692666930616,
     'Generation': 0.17694785326190432}




```python
df_acc_sec_size_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_sec_size_gen.set_index('Generation', inplace=True)
df_acc_sec_size_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.078947</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.060606</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.190476</td>
    </tr>
  </tbody>
</table>
</div>



<a id="imb-sum"></a>

### Summary (Imbalanced)
We summarize and compare our results for the imbalanced dataset as follows:

For the prediction accuracy of different Pokemon types, the best set of features are shown below:


```python
df_prim_types = pd.DataFrame()
df_prim_types['All Features'] = df_acc_prim_all_types['Test Accuracy']
df_prim_types['Color Features'] = df_acc_prim_color_types['Test Accuracy']
df_prim_types['Size Features'] = df_acc_prim_size_types['Test Accuracy']
# df_acc_sec_size_gen
df_prim_types['Best set of features'] = df_prim_types.idxmax(axis=1)
df_prim_types.loc[df_prim_types[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'

df_prim_types['Primary Type'] = df_prim_types.index
df_prim_types.set_index('Primary Type', inplace=True)
df_prim_types
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Primary Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.400000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.333333</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.533333</td>
      <td>0.466667</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.333333</td>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.391304</td>
      <td>0.478261</td>
      <td>0.000000</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.384615</td>
      <td>0.307692</td>
      <td>0.615385</td>
      <td>Size Features</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.500000</td>
      <td>0.607143</td>
      <td>0.500000</td>
      <td>Color Features</td>
    </tr>
  </tbody>
</table>
</div>



For different Pokemon types, it is noticeable how a different combination of features yields different accuracies for the classification. For example, types like Poison are best modeled using color features only due to their distinct violet color while Normal type Pokemon are different to classify using colors and are actually best predicted using size features.


```python
df_prim_gen = pd.DataFrame()
df_prim_gen['All Features'] = df_acc_prim_all_gen['Test Accuracy']
df_prim_gen['Color Features'] = df_acc_prim_color_gen['Test Accuracy']
df_prim_gen['Size Features'] = df_acc_prim_size_gen['Test Accuracy']
# df_acc_sec_size_gen
df_prim_gen['Best set of features'] = df_prim_gen.idxmax(axis=1)
df_prim_gen.loc[df_prim_gen[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'

df_prim_gen['Generation'] = df_prim_gen.index
df_prim_gen.set_index('Generation', inplace=True)
df_prim_gen
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.368421</td>
      <td>0.342105</td>
      <td>0.263158</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.318182</td>
      <td>0.409091</td>
      <td>0.136364</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.181818</td>
      <td>0.212121</td>
      <td>0.181818</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.125000</td>
      <td>0.125000</td>
      <td>0.166667</td>
      <td>Size Features</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.340909</td>
      <td>0.250000</td>
      <td>0.136364</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.238095</td>
      <td>0.190476</td>
      <td>0.047619</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.142857</td>
      <td>0.238095</td>
      <td>0.047619</td>
      <td>Color Features</td>
    </tr>
  </tbody>
</table>
</div>



Across different generations, different types of Pokemon are introduced. This is why there are no significant differences across different generation groups on the accuracies for different set of features.

Now, we summarize the results for classification of Pokemon secondary types.


```python
df_sec_types = pd.DataFrame()
df_sec_types['All Features'] = df_acc_sec_all_types['Test Accuracy']
df_sec_types['Color Features'] = df_acc_sec_color_types['Test Accuracy']
df_sec_types['Size Features'] = df_acc_sec_size_types['Test Accuracy']
# df_acc_sec_size_gen
df_sec_types['Best set of features'] = df_sec_types.idxmax(axis=1)
df_sec_types.loc[df_sec_types[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'


df_sec_types['Secondary Type'] = df_sec_types.index
df_sec_types.set_index('Secondary Type', inplace=True)
df_sec_types
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Secondary Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0.300000</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.555556</td>
      <td>0.444444</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.333333</td>
      <td>0.444444</td>
      <td>0.111111</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.545455</td>
      <td>0.545455</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.315789</td>
      <td>0.263158</td>
      <td>0.368421</td>
      <td>Size Features</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.333333</td>
      <td>0.380952</td>
      <td>0.142857</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.300000</td>
      <td>0.250000</td>
      <td>0.550000</td>
      <td>Size Features</td>
    </tr>
  </tbody>
</table>
</div>



Note that the data here is still imbalanced but similar observation can be made here wherein different set of features yield different accuracies for different Pokemon typings. Interestingly, there are less zero accuracies here than the Primary types, which could be an indicator of the higher predictability of Secondary Types rather than the Primary Types with our features. However, it could also be an artifact of the distribution of classes for the Secondary Types. We will stop that analysis there and balance the data instead to obtain better accuracies.


```python
df_sec_gen = pd.DataFrame()
df_sec_gen['All Features'] = df_acc_sec_all_gen['Test Accuracy']
df_sec_gen['Color Features'] = df_acc_sec_color_gen['Test Accuracy']
df_sec_gen['Size Features'] = df_acc_sec_size_gen['Test Accuracy']
# df_acc_sec_size_gen
df_sec_gen['Best set of features'] = df_sec_gen.idxmax(axis=1)
df_sec_gen.loc[df_sec_gen[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'


df_sec_gen['Generation'] = df_sec_gen.index
df_sec_gen.set_index('Generation', inplace=True)
df_sec_gen
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.289474</td>
      <td>0.263158</td>
      <td>0.078947</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.181818</td>
      <td>0.227273</td>
      <td>0.090909</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.121212</td>
      <td>0.121212</td>
      <td>0.060606</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.333333</td>
      <td>0.250000</td>
      <td>0.166667</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.272727</td>
      <td>0.295455</td>
      <td>0.136364</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.333333</td>
      <td>0.285714</td>
      <td>0.142857</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.095238</td>
      <td>0.047619</td>
      <td>0.190476</td>
      <td>Size Features</td>
    </tr>
  </tbody>
</table>
</div>



Similar to the Primary Types, there are no significant differences in the use of different set of features for the accuracies of Pokemons from different generations.

<a id="balanced"></a>

# Data balancing through undersampling
As mentioned earlier, the data is quite imbalanced. For this analysis, we will use undersampling with two purposes:
1. Make the data balanced
2. Reduce the target classes to improve predictability

Again, we perform this analysis by considering the Primary Type and the Secondary Type as the target separately. The following steps will not be discussed as the methodology employed here will be the same as the <a href='#imbalanced'>imbalanced dataset</a>

<a id="bal-main"></a>

### Primary Type

From the original dataset:


```python
types = df_target_A.value_counts()
plt.barh(types.index[::-1], types.values[::-1])
plt.fill_between([0, 40], [9.5, 9.5], [17.5, 17.5], zorder=2, alpha=0.5)
```




    <matplotlib.collections.PolyCollection at 0x15483b56808>




![png](output_img/output_199_1.png)


The orange filled area will be the sampled data.


```python
thresholded_A = pd.Series(df_target_A.index.values, index=df_target_A)[df_target_A.value_counts()>=40]
df_target_A_balanced = df_target_A[thresholded_A]
```


```python
df_temp = df_target_A_balanced.groupby(df_target_A_balanced).apply(lambda x: x.sample(n=40)).reset_index(level=1)
df_temp.set_index('Pokedex Number', inplace=True)

df_target_A_balanced = df_target_A_balanced[df_temp.index]
```

#### Combined features


```python
df_feat_balanced = df_feat.loc[thresholded_A]
df_feat_balanced = df_feat_balanced.loc[df_temp.index]
```


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat_balanced.astype(float)
y = df_target_A_balanced

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('KNN', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                         weights='uniform'), {'n_neighbors': range(1, 31)})

    Training ('Logistic Regression (L1)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l1',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Logistic Regression (L2)', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=-1, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Linear SVM (L2)', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('NonLinear SVM (RBF)', SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05]), 'gamma': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('NonLinear SVM (Poly)', SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05]), 'gamma': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})

    Training ('Decision Tree', DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best'), {'max_depth': range(1, 11), 'criterion': ['gini', 'entropy']})

    Training ('Random Forest', RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='sqrt',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False), {'max_depth': range(1, 4), 'n_estimators': range(10, 101, 10), 'criterion': ['gini', 'entropy']})

    Training ('GBM', GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='deprecated',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False), {'max_depth': range(1, 4), 'n_estimators': range(10, 101, 10), 'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_bal
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>0.2125</td>
      <td>{'n_neighbors': 6}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression (L1)</td>
      <td>0.3250</td>
      <td>{'C': 3.593813663804626}</td>
      <td>{'mean_r': 1.110367749277645, 'mean_g': 1.1326...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression (L2)</td>
      <td>0.3500</td>
      <td>{'C': 3.593813663804626}</td>
      <td>{'mean_r': 0.9829822026796213, 'mean_g': 1.060...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Linear SVM (L1)</td>
      <td>0.3625</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>{'mean_r': 0.19745981180946412, 'mean_g': 0.17...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Linear SVM (L2)</td>
      <td>0.3375</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>{'mean_r': 0.2656110471293918, 'mean_g': 0.292...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NonLinear SVM (RBF)</td>
      <td>0.3250</td>
      <td>{'C': 599.4842503189421, 'gamma': 0.0001291549...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NonLinear SVM (Poly)</td>
      <td>0.2250</td>
      <td>{'C': 0.0001291549665014884, 'gamma': 3.593813...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Decision Tree</td>
      <td>0.1875</td>
      <td>{'criterion': 'gini', 'max_depth': 8}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Random Forest</td>
      <td>0.3000</td>
      <td>{'criterion': 'gini', 'max_depth': 3, 'n_estim...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GBM</td>
      <td>0.2125</td>
      <td>{'learning_rate': 0.7000000000000001, 'max_dep...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This is the only difference in the analysis above - the model used here is Linear SVM (L1).


```python
coefs = models['Linear SVM (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.05774750035926872, 0.05774750035926872)




![png](output_img/output_211_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'
for i in df_target_A_balanced.value_counts().index[::-1]:
    try:
        top_predictor = X.columns[np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_bal_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Psychic</th>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.272727</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_prim_bal_gen['Top Predictor Variable'][1]
```




    {'mean_r': 0.19745981180946412,
     'mean_g': 0.1765740872076312,
     'mean_b': 0.21061701692350734,
     'mode_r': 0.052611898328098164,
     'mode_g': 0.04165848365759424,
     'mode_b': 0.015984674960662117,
     'mode_freq_r': 0.0480744474970477,
     'mode_freq_g': 0.08566693956256899,
     'mode_freq_b': 0.09761483957488881,
     'std_r': 0.06757341585470647,
     'std_g': 0.09157508111700968,
     'std_b': 0.035739208525317916,
     'png': 0.04774750035926872,
     'size': 0.058394945490497716,
     'mean_width': 0.06075258509166301,
     'mode_width': 0.06956230811534592,
     'std_width': 0.01676515234382168,
     'mean_length': 0.05566867868992349,
     'mode_length': 0.0532176930137803,
     'std_length': 0.07720485213119227,
     'Generation': 0.07540646664154303}




```python
df_acc_prim_bal_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_prim_bal_gen.set_index('Generation', inplace=True)
df_acc_prim_bal_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.409091</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.214286</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>



#### Color features only


```python
df_feat_color_balanced = df_feat_color.loc[thresholded_A]
df_feat_color_balanced = df_feat_color_balanced.loc[df_temp.index]
```


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type1</th>
      <th>Type2</th>
      <th>Generation</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>Fire</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>Fire</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df_feat_color_balanced
X['Generation'] = df.loc[thresholded_A]['Generation'].astype(float)
y = df_target_A_balanced

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[3:4]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_color = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_color.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_bal_color
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear SVM (L1)</td>
      <td>0.375</td>
      <td>{'C': 3.593813663804626}</td>
      <td>mean_b</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Linear SVM (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_color.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.06593061812005134, 0.06593061812005134)




![png](output_img/output_225_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_color_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'
for i in df_target_A_balanced.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_color_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_bal_color_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Psychic</th>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.454545</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_color_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_color_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_prim_bal_color_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_prim_bal_color_gen.set_index('Generation', inplace=True)
df_acc_prim_bal_color_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.409091</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>



#### Size features only


```python
df_feat_size_balanced = df_feat_size.loc[thresholded_A]
df_feat_size_balanced = df_feat_size_balanced.loc[df_temp.index]
```


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat_size_balanced
X['Generation'] = df.loc[thresholded_A]['Generation'].astype(float)
y = df_target_A_balanced

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[3:4]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_size = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_size.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_bal_size
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear SVM (L1)</td>
      <td>0.0875</td>
      <td>{'C': 0.021544346900318846}</td>
      <td>mean_length</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Linear SVM (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_size.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.011650163141296946, 0.011650163141296946)




![png](output_img/output_237_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_size_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'
for i in df_target_A_balanced.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_size_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_prim_bal_size_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Psychic</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_prim_bal_size_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_prim_bal_size_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_prim_bal_size_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_prim_bal_size_gen.set_index('Generation', inplace=True)
df_acc_prim_bal_size_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>



<a id="bal-sec"></a>

### Secondary Type


```python
types = df_target_B.value_counts()
plt.barh(types.index[::-1], types.values[::-1])

plt.fill_between([0, 45], [9.5, 9.5], [17.5, 17.5], zorder=2, alpha=0.5)
```




    <matplotlib.collections.PolyCollection at 0x15483b59a08>




![png](output_img/output_243_1.png)



```python
thresholded_B = pd.Series(df_target_B.index.values, index=df_target_B)[df_target_B.value_counts()>45]
df_target_B_balanced = df_target_B[thresholded_B]
```


```python
df_temp = df_target_B_balanced.groupby(df_target_B_balanced).apply(lambda x: x.sample(n=40)).reset_index(level=1)
df_temp.set_index('Pokedex Number', inplace=True)

df_target_B_balanced = df_target_B_balanced.loc[df_temp.index]
```

#### Combined features


```python
df_feat_balanced_B = df_feat.loc[thresholded_B]
df_feat_balanced_B = df_feat_balanced_B.loc[df_temp.index]
```


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat_balanced_B.astype(float)
y = df_target_B_balanced

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[3:4]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_bal
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear SVM (L1)</td>
      <td>0.3</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>mean_g</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Linear SVM (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.052113408048545876, 0.052113408048545876)




![png](output_img/output_253_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'
for i in df_target_B_balanced.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_bal_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fighting</th>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.142857</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_sec_bal_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_sec_bal_gen.set_index('Generation', inplace=True)
df_acc_sec_bal_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.307692</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.111111</td>
    </tr>
  </tbody>
</table>
</div>



#### Color features only


```python
df_feat_color_balanced_B = df_feat_color.loc[thresholded_B]
df_feat_color_balanced_B = df_feat_color_balanced_B.loc[df_temp.index]
```


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type1</th>
      <th>Type2</th>
      <th>Generation</th>
    </tr>
    <tr>
      <th>Pokedex Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>Fire</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>Fire</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df_feat_color_balanced_B
X['Generation'] = df.loc[thresholded_B]['Generation'].astype(float)
y = df_target_B_balanced

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[3:4]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_color = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_color.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_bal_color
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear SVM (L1)</td>
      <td>0.2625</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>mean_g</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Linear SVM (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_color.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.0367333306606361, 0.0367333306606361)




![png](output_img/output_266_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_color_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'
for i in df_target_B_balanced.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_color_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_bal_color_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fighting</th>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.357143</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_color_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_color_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_sec_bal_color_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_sec_bal_color_gen.set_index('Generation', inplace=True)
df_acc_sec_bal_color_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.277778</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.307692</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Size features only


```python
df_feat_size_balanced_B = df_feat_size.loc[thresholded_B]
df_feat_size_balanced_B = df_feat_size_balanced_B.loc[df_temp.index]
```


```python
cl1 = KNeighborsClassifier(n_jobs=-1)
cl2 = LogisticRegression(penalty='l1', max_iter=1000,
                         solver='liblinear', n_jobs=-1)
cl3 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
cl4 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
cl5 = LinearSVC(penalty='l2', max_iter=10000)
cl6 = SVC(kernel='rbf', )
cl7 = SVC(kernel='poly', degree=3)
cl8 = DecisionTreeClassifier()
cl9 = RandomForestClassifier(max_features='sqrt')
cl10 = GradientBoostingClassifier()
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)

estimators = [('KNN', cl1, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', cl2, {'C':C_list}),
              ('Logistic Regression (L2)', cl3, {'C':C_list}),
              ('Linear SVM (L1)', cl4, {'C':C_list}),
              ('Linear SVM (L2)', cl5, {'C':C_list}),
              ('NonLinear SVM (RBF)', cl6, {'C':C_list,
                                            'gamma':gamma_list}),
              ('NonLinear SVM (Poly)', cl7, {'C':C_list,
                                             'gamma':gamma_list}),
              ('Decision Tree', cl8, {'max_depth':range(1,11),
                                      'criterion':['gini', 'entropy']}),
              ('Random Forest', cl9, {'max_depth':range(1,4),
                                      'n_estimators':range(10,101,10),
                                      'criterion':['gini', 'entropy']}),
              ('GBM', cl10, {'max_depth':range(1,4),
                            'n_estimators':range(10,101,10),
                            'learning_rate':np.arange(0.1,1.01,0.1)})]
```


```python
X = df_feat_size_balanced_B
X['Generation'] = df.loc[thresholded_B]['Generation'].astype(float)
y = df_target_B_balanced

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                            random_state=1)
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```


```python
models = {}
for est in estimators[3:4]:
    print(f'Training {est}\n')
    gs_cv = model_selection.GridSearchCV(est[1], param_grid=est[2], n_jobs=4)
    gs_cv.fit(X_train, y_train)
    models[est[0]] = gs_cv
```

    Training ('Linear SVM (L1)', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=10000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
              verbose=0), {'C': array([1.00000000e-05, 1.29154967e-04, 1.66810054e-03, 2.15443469e-02,
           2.78255940e-01, 3.59381366e+00, 4.64158883e+01, 5.99484250e+02,
           7.74263683e+03, 1.00000000e+05])})




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_size = pd.DataFrame(columns=cols)

for i, m in enumerate(models):

    try:
        top_predictor = X.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_).mean(axis=0))]

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_size.loc[i] = [m,
                 models[m].best_estimator_.score(X_val, y_val),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_bal_size
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear SVM (L1)</td>
      <td>0.175</td>
      <td>{'C': 3.593813663804626}</td>
      <td>size</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefs = models['Linear SVM (L1)'].best_estimator_.coef_.mean(axis=0)
plt.barh(df_feat_size.columns[np.argsort(coefs)], np.sort(coefs))
plt.xlim(-max(abs(coefs)) -0.01, max(abs(coefs))+0.01)
```




    (-0.034051332746705916, 0.034051332746705916)




![png](output_img/output_278_1.png)



```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_size_types = pd.DataFrame(columns=cols)

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'
for i in df_target_B_balanced.value_counts().index[::-1]:
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_size_types.loc[i] = [m,
                 models[m].best_estimator_.score(X_val[y_val==i], y_val[y_val==i]),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_acc_sec_bal_size_types[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fighting</th>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.285714</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_acc_sec_bal_size_gen = pd.DataFrame(columns=cols)
gen_scaled = dict(zip(range(1, 8), np.unique(X_val[:, -1])))

# for i, m in enumerate(models):
m = 'Linear SVM (L1)'

for i in range(1,8):
    try:
        top_predictor = dict(zip(X.columns, np.abs(models[m].best_estimator_.coef_).mean(axis=0)))

    except AttributeError:
        top_predictor = np.nan

    df_acc_sec_bal_size_gen.loc[i] = [m,
                     models[m].best_estimator_.score(X_val[X_val[:, -1]==gen_scaled[i]], y_val[X_val[:, -1]==gen_scaled[i]]),
                     models[m].best_params_ ,
                     top_predictor]
```


```python
df_acc_sec_bal_size_gen['Generation'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
df_acc_sec_bal_size_gen.set_index('Generation', inplace=True)
df_acc_sec_bal_size_gen[['Test Accuracy']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Accuracy</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.153846</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>



<a id="bal-sum"></a>

### Summary (Balanced)
We summarize and compare our results for the balanced dataset as follows:

For the prediction accuracy of different Pokemon types, the best set of features are shown below:


```python
df_prim_bal_types = pd.DataFrame()
df_prim_bal_types['All Features'] = df_acc_prim_bal_types['Test Accuracy']
df_prim_bal_types['Color Features'] = df_acc_prim_bal_color_types['Test Accuracy']
df_prim_bal_types['Size Features'] = df_acc_prim_bal_size_types['Test Accuracy']
# df_acc_sec_size_gen
df_prim_bal_types['Best set of features'] = df_prim_bal_types.idxmax(axis=1)
df_prim_bal_types.loc[df_prim_bal_types[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'

df_prim_bal_types['Primary Type'] = df_prim_bal_types.index
df_prim_bal_types.set_index('Primary Type', inplace=True)
df_prim_bal_types
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Primary Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Psychic</th>
      <td>0.250000</td>
      <td>0.125000</td>
      <td>0.0</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.444444</td>
      <td>0.555556</td>
      <td>0.0</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>1.0</td>
      <td>Size Features</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0.833333</td>
      <td>0.750000</td>
      <td>0.0</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>0.272727</td>
      <td>0.454545</td>
      <td>0.0</td>
      <td>Color Features</td>
    </tr>
  </tbody>
</table>
</div>



Now that the dataset is balanced, our analysis will be more complete since the accuracies will not be biased towards types with higher population. It is noticeable how the best features are mostly All Features with the exception of bug types which is more predictable with only the size features. Note that the most predictable types are water, fire, and grass, mainly due to the color features.


```python
df_prim_bal_gen = pd.DataFrame()
df_prim_bal_gen['All Features'] = df_acc_prim_bal_gen['Test Accuracy']
df_prim_bal_gen['Color Features'] = df_acc_prim_bal_color_gen['Test Accuracy']
df_prim_bal_gen['Size Features'] = df_acc_prim_bal_size_gen['Test Accuracy']
# df_acc_sec_size_gen
df_prim_bal_gen['Best set of features'] = df_prim_bal_gen.idxmax(axis=1)
df_prim_bal_gen.loc[df_prim_bal_gen[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'

df_prim_bal_gen['Generation'] = df_prim_bal_gen.index
df_prim_bal_gen.set_index('Generation', inplace=True)
df_prim_bal_gen
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.409091</td>
      <td>0.409091</td>
      <td>0.090909</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.416667</td>
      <td>0.500000</td>
      <td>0.083333</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.416667</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.214286</td>
      <td>0.285714</td>
      <td>0.071429</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.400000</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>All Features</td>
    </tr>
  </tbody>
</table>
</div>



For different generations, it can be seen that, again, All Features generally yield high accuracy except for Gen I and Gen III where color features are better predictors.


```python
df_sec_bal_types = pd.DataFrame()
df_sec_bal_types['All Features'] = df_acc_sec_bal_types['Test Accuracy']
df_sec_bal_types['Color Features'] = df_acc_sec_bal_color_types['Test Accuracy']
df_sec_bal_types['Size Features'] = df_acc_sec_bal_size_types['Test Accuracy']
# df_acc_sec_bal_size_gen
df_sec_bal_types['Best set of features'] = df_sec_bal_types.idxmax(axis=1)
df_sec_bal_types.loc[df_sec_bal_types[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'


df_sec_bal_types['Secondary Type'] = df_sec_bal_types.index
df_sec_bal_types.set_index('Secondary Type', inplace=True)
df_sec_bal_types
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Secondary Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fighting</th>
      <td>0.222222</td>
      <td>0.111111</td>
      <td>0.222222</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.111111</td>
      <td>0.222222</td>
      <td>0.000000</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.333333</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.375000</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.428571</td>
      <td>0.357143</td>
      <td>0.142857</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>Size Features</td>
    </tr>
  </tbody>
</table>
</div>



For the secondary types, it is noteworthy how Flying types are the ones that are very predictable with grass and poison being next. The flying types are predictable due to the combination of the features while grass and poison are due to their color features.


```python
df_sec_bal_gen = pd.DataFrame()
df_sec_bal_gen['All Features'] = df_acc_sec_bal_gen['Test Accuracy']
df_sec_bal_gen['Color Features'] = df_acc_sec_bal_color_gen['Test Accuracy']
df_sec_bal_gen['Size Features'] = df_acc_sec_bal_size_gen['Test Accuracy']
# df_acc_sec_bal_size_gen
df_sec_bal_gen['Best set of features'] = df_sec_bal_gen.idxmax(axis=1)
df_sec_bal_gen.loc[df_sec_bal_gen[['All Features', 'Color Features', 'Size Features']].sum(axis=1)==0, 'Best set of features'] = 'None of the above'


df_sec_bal_gen['Generation'] = df_sec_bal_gen.index
df_sec_bal_gen.set_index('Generation', inplace=True)
df_sec_bal_gen
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>All Features</th>
      <th>Color Features</th>
      <th>Size Features</th>
      <th>Best set of features</th>
    </tr>
    <tr>
      <th>Generation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>I</th>
      <td>0.222222</td>
      <td>0.277778</td>
      <td>0.222222</td>
      <td>Color Features</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.307692</td>
      <td>0.307692</td>
      <td>0.153846</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.333333</td>
      <td>0.250000</td>
      <td>0.083333</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>IV</th>
      <td>0.416667</td>
      <td>0.416667</td>
      <td>0.083333</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.500000</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>All Features</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None of the above</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.222222</td>
      <td>Size Features</td>
    </tr>
  </tbody>
</table>
</div>



For the different generations, again, it is consistent how the predictability of a type is a combination of the color feature and the size feature.

In the end, what can be observed is that the generation only has minimal effect on the predictability of the Pokemon types, regardless of primary or secondary. On the other hand, there are Pokemon types that are more dependent on the color features like water, fire, and grass while there are those that are more dependent on size features like bug and flying.

<a id="conclusion"></a>

# Conclusion

Images are a treasure trove of data. However, this also necessitates a computational capability to process this large data. As such, depending on the purposes of classification, it may be sufficient to compare extracted features from the image rather than the pixels, drastically reducing the number of features in the model. Case in point is the Pokemon dataset where if we are only to classify Fire, Water, or Grass, it is sufficient to use color features. This finding extends to other kinds of dataset that has too many features, wherein summary statistics may be enough as features. This leads to better interpretability and faster runtime while only sacrificing a little accuracy.

Another finding is different combination of features yields to different accuracies. Increasing the features can only introduce noise, thereby decreasing accuracy. That is, more features does not translate to higher accuracy.

Breaking down the accuracies to the different classes can give you an idea whether some features are still needed, or more specifically, whether the features can only predict certain classes.

In order to improve the results, the data must be increased through oversampling (SMOTE) or by generating new images through image manipulation techniques. Another recommendation is to use other color spaces aside from RGB like HSV.


<a id="ref"></a>
# References

1. https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types#images.zip
2. https://bulbapedia.bulbagarden.net/


<a id="ack"></a>
# Acknowledgement

I would like to acknowledge:
- Vishal Subbiah for the Pokemon Images dataset in Kaggle.
- LT13 for the help during the project
- MSDS professors and classmates

<a id="appendix"></a>

# Appendix

## Generate some images
The code below is for generating some of the images in the presentation


```python
# im = Image.open(f'images/pikachu.png').convert('RGBA')
# im_arr = np.array(im)

# plt.figure()
# plt.imshow(im_arr[:,:,0], cmap='Reds')
# plt.axis('off')
# # plt.savefig('red_pikachu.png', dpi=300, bbox_inches='tight')
# plt.close()
# plt.figure()
# plt.imshow(im_arr[:,:,1], cmap='Greens')
# plt.axis('off')
# # plt.savefig('green_pikachu.png', dpi=300, bbox_inches='tight')
# plt.close()
# plt.figure()
# plt.imshow(im_arr[:,:,2], cmap='Blues')
# plt.axis('off')
# # plt.savefig('blue_pikachu.png', dpi=300, bbox_inches='tight')
# plt.close()
# plt.figure()
# plt.imshow(im_arr)
# plt.axis('off')
# # plt.savefig('pikachu.png', dpi=300, bbox_inches='tight')
# # plt.close()
```
