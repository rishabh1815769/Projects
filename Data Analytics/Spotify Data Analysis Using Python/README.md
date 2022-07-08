# Spotify Data Analysis Using Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
df_tracks = pd.read_csv('C:/Users/risha/OneDrive/Documents/Data Analytics and Data Engineering Projects/Spotify Data Analysis Using Python/Datasets/tracks.csv')
```

```python
df_tracks.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>artists</th>
      <th>id_artists</th>
      <th>release_date</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35iwgR4jXetI318WEWsa1Q</td>
      <td>Carve</td>
      <td>6</td>
      <td>126903</td>
      <td>0</td>
      <td>['Uli']</td>
      <td>['45tIt06XoI0Iio4LBEVpls']</td>
      <td>1922-02-22</td>
      <td>0.645</td>
      <td>0.4450</td>
      <td>0</td>
      <td>-13.338</td>
      <td>1</td>
      <td>0.4510</td>
      <td>0.674</td>
      <td>0.7440</td>
      <td>0.151</td>
      <td>0.127</td>
      <td>104.851</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>021ht4sdgPcrDgSk7JTbKY</td>
      <td>Capítulo 2.16 - Banquero Anarquista</td>
      <td>0</td>
      <td>98200</td>
      <td>0</td>
      <td>['Fernando Pessoa']</td>
      <td>['14jtPCOoNZwquk5wd9DxrY']</td>
      <td>1922-06-01</td>
      <td>0.695</td>
      <td>0.2630</td>
      <td>0</td>
      <td>-22.136</td>
      <td>1</td>
      <td>0.9570</td>
      <td>0.797</td>
      <td>0.0000</td>
      <td>0.148</td>
      <td>0.655</td>
      <td>102.009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>07A5yehtSnoedViJAZkNnc</td>
      <td>Vivo para Quererte - Remasterizado</td>
      <td>0</td>
      <td>181640</td>
      <td>0</td>
      <td>['Ignacio Corsini']</td>
      <td>['5LiOoJbxVSAMkBS2fUm3X2']</td>
      <td>1922-03-21</td>
      <td>0.434</td>
      <td>0.1770</td>
      <td>1</td>
      <td>-21.180</td>
      <td>1</td>
      <td>0.0512</td>
      <td>0.994</td>
      <td>0.0218</td>
      <td>0.212</td>
      <td>0.457</td>
      <td>130.418</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>08FmqUhxtyLTn6pAh6bk45</td>
      <td>El Prisionero - Remasterizado</td>
      <td>0</td>
      <td>176907</td>
      <td>0</td>
      <td>['Ignacio Corsini']</td>
      <td>['5LiOoJbxVSAMkBS2fUm3X2']</td>
      <td>1922-03-21</td>
      <td>0.321</td>
      <td>0.0946</td>
      <td>7</td>
      <td>-27.961</td>
      <td>1</td>
      <td>0.0504</td>
      <td>0.995</td>
      <td>0.9180</td>
      <td>0.104</td>
      <td>0.397</td>
      <td>169.980</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>08y9GfoqCWfOGsKdwojr5e</td>
      <td>Lady of the Evening</td>
      <td>0</td>
      <td>163080</td>
      <td>0</td>
      <td>['Dick Haymes']</td>
      <td>['3BiJGZsyX9sJchTqcSA7Su']</td>
      <td>1922</td>
      <td>0.402</td>
      <td>0.1580</td>
      <td>3</td>
      <td>-16.900</td>
      <td>0</td>
      <td>0.0390</td>
      <td>0.989</td>
      <td>0.1300</td>
      <td>0.311</td>
      <td>0.196</td>
      <td>103.220</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Finding out number of null values in the dataset
pd.isnull(df_tracks).sum()
```

    id                   0
    name                71
    popularity           0
    duration_ms          0
    explicit             0
    artists              0
    id_artists           0
    release_date         0
    danceability         0
    energy               0
    key                  0
    loudness             0
    mode                 0
    speechiness          0
    acousticness         0
    instrumentalness     0
    liveness             0
    valence              0
    tempo                0
    time_signature       0
    dtype: int64

```python
# Getting information about contents of dataset
df_tracks.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 586672 entries, 0 to 586671
    Data columns (total 20 columns):
     #   Column            Non-Null Count   Dtype
    ---  ------            --------------   -----
     0   id                586672 non-null  object
     1   name              586601 non-null  object
     2   popularity        586672 non-null  int64
     3   duration_ms       586672 non-null  int64
     4   explicit          586672 non-null  int64
     5   artists           586672 non-null  object
     6   id_artists        586672 non-null  object
     7   release_date      586672 non-null  object
     8   danceability      586672 non-null  float64
     9   energy            586672 non-null  float64
     10  key               586672 non-null  int64
     11  loudness          586672 non-null  float64
     12  mode              586672 non-null  int64
     13  speechiness       586672 non-null  float64
     14  acousticness      586672 non-null  float64
     15  instrumentalness  586672 non-null  float64
     16  liveness          586672 non-null  float64
     17  valence           586672 non-null  float64
     18  tempo             586672 non-null  float64
     19  time_signature    586672 non-null  int64
    dtypes: float64(9), int64(6), object(5)
    memory usage: 89.5+ MB

```python
# Finding the least popular songs
sorted_df = df_tracks.sort_values('popularity', ascending=True).head(10)
sorted_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>artists</th>
      <th>id_artists</th>
      <th>release_date</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>546130</th>
      <td>181rTRhCcggZPwP2TUcVqm</td>
      <td>Newspaper Reports On Abner, 20 February 1935</td>
      <td>0</td>
      <td>896575</td>
      <td>0</td>
      <td>['Norris Goff', 'Chester Lauck', 'Carlton Bric...</td>
      <td>['3WCwCPDMpGzrt0Qz6quumy', '7vk8UqABg0Sga78GI3...</td>
      <td>1935-02-20</td>
      <td>0.595</td>
      <td>0.262</td>
      <td>8</td>
      <td>-17.746</td>
      <td>1</td>
      <td>0.9320</td>
      <td>0.993</td>
      <td>0.007510</td>
      <td>0.0991</td>
      <td>0.320</td>
      <td>79.849</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546222</th>
      <td>0yOCz3V5KMm8l1T8EFc60i</td>
      <td>恋は水の上で</td>
      <td>0</td>
      <td>188440</td>
      <td>0</td>
      <td>['Hibari Misora']</td>
      <td>['1m5pMY5blqJwdxJ7vqQtuN']</td>
      <td>1949</td>
      <td>0.418</td>
      <td>0.388</td>
      <td>0</td>
      <td>-8.580</td>
      <td>1</td>
      <td>0.0358</td>
      <td>0.925</td>
      <td>0.000014</td>
      <td>0.1050</td>
      <td>0.439</td>
      <td>94.549</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546221</th>
      <td>0y48Hhwe52099UqYjegRCO</td>
      <td>私の誕生日</td>
      <td>0</td>
      <td>173467</td>
      <td>0</td>
      <td>['Hibari Misora']</td>
      <td>['1m5pMY5blqJwdxJ7vqQtuN']</td>
      <td>1949</td>
      <td>0.642</td>
      <td>0.178</td>
      <td>5</td>
      <td>-11.700</td>
      <td>1</td>
      <td>0.0501</td>
      <td>0.993</td>
      <td>0.000943</td>
      <td>0.0928</td>
      <td>0.715</td>
      <td>119.013</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546220</th>
      <td>0xCmgtf9ka07hkZg3D6PaV</td>
      <td>エル・チョクロ (EL CHOCLO)</td>
      <td>0</td>
      <td>205280</td>
      <td>0</td>
      <td>['Hibari Misora']</td>
      <td>['1m5pMY5blqJwdxJ7vqQtuN']</td>
      <td>1949</td>
      <td>0.695</td>
      <td>0.467</td>
      <td>0</td>
      <td>-12.236</td>
      <td>0</td>
      <td>0.0422</td>
      <td>0.827</td>
      <td>0.000000</td>
      <td>0.0861</td>
      <td>0.756</td>
      <td>125.941</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546219</th>
      <td>0tBXS3VuCPX7KWUFH2nros</td>
      <td>恋は不思議なもの</td>
      <td>0</td>
      <td>185733</td>
      <td>0</td>
      <td>['Hibari Misora']</td>
      <td>['1m5pMY5blqJwdxJ7vqQtuN']</td>
      <td>1949</td>
      <td>0.389</td>
      <td>0.388</td>
      <td>2</td>
      <td>-8.221</td>
      <td>1</td>
      <td>0.0351</td>
      <td>0.869</td>
      <td>0.000000</td>
      <td>0.0924</td>
      <td>0.372</td>
      <td>72.800</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546218</th>
      <td>0qrKnQtYDVJhKFAXTHYVS9</td>
      <td>ゆうべはどうしたの (WHATSA MALLA U)</td>
      <td>0</td>
      <td>183427</td>
      <td>0</td>
      <td>['Hibari Misora']</td>
      <td>['1m5pMY5blqJwdxJ7vqQtuN']</td>
      <td>1949</td>
      <td>0.631</td>
      <td>0.249</td>
      <td>5</td>
      <td>-11.883</td>
      <td>1</td>
      <td>0.0355</td>
      <td>0.951</td>
      <td>0.000000</td>
      <td>0.0814</td>
      <td>0.517</td>
      <td>131.097</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546217</th>
      <td>0nqsDxOeKSwEzp3AUQAAqS</td>
      <td>Screen Director's Playhouse, Music For Million...</td>
      <td>0</td>
      <td>1767071</td>
      <td>0</td>
      <td>['Wilms Herbert', 'June Allyson', 'Joseph Kear...</td>
      <td>['2rbm8QWvmnVwxFo84EVM1h', '4yW5adMgyIfHFzaL9i...</td>
      <td>1949-04-10</td>
      <td>0.533</td>
      <td>0.317</td>
      <td>7</td>
      <td>-13.047</td>
      <td>1</td>
      <td>0.9180</td>
      <td>0.682</td>
      <td>0.000000</td>
      <td>0.3330</td>
      <td>0.336</td>
      <td>76.836</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546216</th>
      <td>0kGEdsxVLYjCdfxM9tbezd</td>
      <td>ブルーマンボ</td>
      <td>0</td>
      <td>162147</td>
      <td>0</td>
      <td>['Hibari Misora']</td>
      <td>['1m5pMY5blqJwdxJ7vqQtuN']</td>
      <td>1949</td>
      <td>0.529</td>
      <td>0.546</td>
      <td>0</td>
      <td>-6.462</td>
      <td>0</td>
      <td>0.0418</td>
      <td>0.784</td>
      <td>0.000000</td>
      <td>0.3750</td>
      <td>0.903</td>
      <td>128.604</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546215</th>
      <td>0bc3PUZurUUXrY7yqoOxjq</td>
      <td>Screen Director's Playhouse, Trade Winds direc...</td>
      <td>0</td>
      <td>1776652</td>
      <td>0</td>
      <td>['Wally Maher', 'Tay Garnett', 'Lurene Tuttle'...</td>
      <td>['7hkhJTTI3VnUGVWUt8SJXT', '3kYeeIpRCgJz4fQYDv...</td>
      <td>1949-05-19</td>
      <td>0.599</td>
      <td>0.321</td>
      <td>0</td>
      <td>-15.428</td>
      <td>0</td>
      <td>0.9330</td>
      <td>0.808</td>
      <td>0.000000</td>
      <td>0.5570</td>
      <td>0.379</td>
      <td>93.025</td>
      <td>4</td>
    </tr>
    <tr>
      <th>546214</th>
      <td>0Wwm0ruSjYMIiWG0nyAI1F</td>
      <td>Screen Director's Playhouse, It's A Wonderful ...</td>
      <td>0</td>
      <td>1767576</td>
      <td>0</td>
      <td>['Joseph Granby', 'Jimmy Stewart', 'Irene Tedr...</td>
      <td>['6GK59BC4LJzqR0OpHAX2S3', '58BzBaExrnrx898sby...</td>
      <td>1949-05-08</td>
      <td>0.645</td>
      <td>0.341</td>
      <td>8</td>
      <td>-12.177</td>
      <td>1</td>
      <td>0.8670</td>
      <td>0.690</td>
      <td>0.000000</td>
      <td>0.1530</td>
      <td>0.431</td>
      <td>117.591</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 10 Most popular songs having popularity more than 90
most_popular = df_tracks.query('popularity>90', inplace=False).sort_values('popularity', ascending = False)
most_popular[:10]
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>artists</th>
      <th>id_artists</th>
      <th>release_date</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93802</th>
      <td>4iJyoBOLtHqaGxP12qzhQI</td>
      <td>Peaches (feat. Daniel Caesar &amp; Giveon)</td>
      <td>100</td>
      <td>198082</td>
      <td>1</td>
      <td>['Justin Bieber', 'Daniel Caesar', 'Giveon']</td>
      <td>['1uNFoZAHBGtllmzznpCI3s', '20wkVLutqVOYrc0kxF...</td>
      <td>2021-03-19</td>
      <td>0.677</td>
      <td>0.696</td>
      <td>0</td>
      <td>-6.181</td>
      <td>1</td>
      <td>0.1190</td>
      <td>0.32100</td>
      <td>0.000000</td>
      <td>0.4200</td>
      <td>0.464</td>
      <td>90.030</td>
      <td>4</td>
    </tr>
    <tr>
      <th>93803</th>
      <td>7lPN2DXiMsVn7XUKtOW1CS</td>
      <td>drivers license</td>
      <td>99</td>
      <td>242014</td>
      <td>1</td>
      <td>['Olivia Rodrigo']</td>
      <td>['1McMsnEElThX1knmY4oliG']</td>
      <td>2021-01-08</td>
      <td>0.585</td>
      <td>0.436</td>
      <td>10</td>
      <td>-8.761</td>
      <td>1</td>
      <td>0.0601</td>
      <td>0.72100</td>
      <td>0.000013</td>
      <td>0.1050</td>
      <td>0.132</td>
      <td>143.874</td>
      <td>4</td>
    </tr>
    <tr>
      <th>93804</th>
      <td>3Ofmpyhv5UAQ70mENzB277</td>
      <td>Astronaut In The Ocean</td>
      <td>98</td>
      <td>132780</td>
      <td>0</td>
      <td>['Masked Wolf']</td>
      <td>['1uU7g3DNSbsu0QjSEqZtEd']</td>
      <td>2021-01-06</td>
      <td>0.778</td>
      <td>0.695</td>
      <td>4</td>
      <td>-6.865</td>
      <td>0</td>
      <td>0.0913</td>
      <td>0.17500</td>
      <td>0.000000</td>
      <td>0.1500</td>
      <td>0.472</td>
      <td>149.996</td>
      <td>4</td>
    </tr>
    <tr>
      <th>92810</th>
      <td>5QO79kh1waicV47BqGRL3g</td>
      <td>Save Your Tears</td>
      <td>97</td>
      <td>215627</td>
      <td>1</td>
      <td>['The Weeknd']</td>
      <td>['1Xyo4u8uXC1ZmMpatF05PJ']</td>
      <td>2020-03-20</td>
      <td>0.680</td>
      <td>0.826</td>
      <td>0</td>
      <td>-5.487</td>
      <td>1</td>
      <td>0.0309</td>
      <td>0.02120</td>
      <td>0.000012</td>
      <td>0.5430</td>
      <td>0.644</td>
      <td>118.051</td>
      <td>4</td>
    </tr>
    <tr>
      <th>92811</th>
      <td>6tDDoYIxWvMLTdKpjFkc1B</td>
      <td>telepatía</td>
      <td>97</td>
      <td>160191</td>
      <td>0</td>
      <td>['Kali Uchis']</td>
      <td>['1U1el3k54VvEUzo3ybLPlM']</td>
      <td>2020-12-04</td>
      <td>0.653</td>
      <td>0.524</td>
      <td>11</td>
      <td>-9.016</td>
      <td>0</td>
      <td>0.0502</td>
      <td>0.11200</td>
      <td>0.000000</td>
      <td>0.2030</td>
      <td>0.553</td>
      <td>83.970</td>
      <td>4</td>
    </tr>
    <tr>
      <th>92813</th>
      <td>0VjIjW4GlUZAMYd2vXMi3b</td>
      <td>Blinding Lights</td>
      <td>96</td>
      <td>200040</td>
      <td>0</td>
      <td>['The Weeknd']</td>
      <td>['1Xyo4u8uXC1ZmMpatF05PJ']</td>
      <td>2020-03-20</td>
      <td>0.514</td>
      <td>0.730</td>
      <td>1</td>
      <td>-5.934</td>
      <td>1</td>
      <td>0.0598</td>
      <td>0.00146</td>
      <td>0.000095</td>
      <td>0.0897</td>
      <td>0.334</td>
      <td>171.005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>93805</th>
      <td>7MAibcTli4IisCtbHKrGMh</td>
      <td>Leave The Door Open</td>
      <td>96</td>
      <td>242096</td>
      <td>0</td>
      <td>['Bruno Mars', 'Anderson .Paak', 'Silk Sonic']</td>
      <td>['0du5cEVh5yTK9QJze8zA0C', '3jK9MiCrA42lLAdMGU...</td>
      <td>2021-03-05</td>
      <td>0.586</td>
      <td>0.616</td>
      <td>5</td>
      <td>-7.964</td>
      <td>1</td>
      <td>0.0324</td>
      <td>0.18200</td>
      <td>0.000000</td>
      <td>0.0927</td>
      <td>0.719</td>
      <td>148.088</td>
      <td>4</td>
    </tr>
    <tr>
      <th>92814</th>
      <td>6f3Slt0GbA2bPZlz0aIFXN</td>
      <td>The Business</td>
      <td>95</td>
      <td>164000</td>
      <td>0</td>
      <td>['Tiësto']</td>
      <td>['2o5jDhtHVPhrJdv3cEQ99Z']</td>
      <td>2020-09-16</td>
      <td>0.798</td>
      <td>0.620</td>
      <td>8</td>
      <td>-7.079</td>
      <td>0</td>
      <td>0.2320</td>
      <td>0.41400</td>
      <td>0.019200</td>
      <td>0.1120</td>
      <td>0.235</td>
      <td>120.031</td>
      <td>4</td>
    </tr>
    <tr>
      <th>91866</th>
      <td>60ynsPSSKe6O3sfwRnIBRf</td>
      <td>Streets</td>
      <td>94</td>
      <td>226987</td>
      <td>1</td>
      <td>['Doja Cat']</td>
      <td>['5cj0lLjcoR7YOSnhnX0Po5']</td>
      <td>2019-11-07</td>
      <td>0.749</td>
      <td>0.463</td>
      <td>11</td>
      <td>-8.433</td>
      <td>1</td>
      <td>0.0828</td>
      <td>0.20800</td>
      <td>0.037100</td>
      <td>0.3370</td>
      <td>0.190</td>
      <td>90.028</td>
      <td>4</td>
    </tr>
    <tr>
      <th>92816</th>
      <td>3FAJ6O0NOHQV8Mc5Ri6ENp</td>
      <td>Heartbreak Anniversary</td>
      <td>94</td>
      <td>198371</td>
      <td>0</td>
      <td>['Giveon']</td>
      <td>['4fxd5Ee7UefO4CUXgwJ7IP']</td>
      <td>2020-03-27</td>
      <td>0.449</td>
      <td>0.465</td>
      <td>0</td>
      <td>-8.964</td>
      <td>1</td>
      <td>0.0791</td>
      <td>0.52400</td>
      <td>0.000001</td>
      <td>0.3030</td>
      <td>0.543</td>
      <td>89.087</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Change index to release date

df_tracks.set_index("release_date", inplace=True)
df_tracks.index = pd.to_datetime(df_tracks.index)
df_tracks.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>artists</th>
      <th>id_artists</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
    </tr>
    <tr>
      <th>release_date</th>
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
      <th>1922-02-22</th>
      <td>35iwgR4jXetI318WEWsa1Q</td>
      <td>Carve</td>
      <td>6</td>
      <td>126903</td>
      <td>0</td>
      <td>['Uli']</td>
      <td>['45tIt06XoI0Iio4LBEVpls']</td>
      <td>0.645</td>
      <td>0.4450</td>
      <td>0</td>
      <td>-13.338</td>
      <td>1</td>
      <td>0.4510</td>
      <td>0.674</td>
      <td>0.7440</td>
      <td>0.151</td>
      <td>0.127</td>
      <td>104.851</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1922-06-01</th>
      <td>021ht4sdgPcrDgSk7JTbKY</td>
      <td>Capítulo 2.16 - Banquero Anarquista</td>
      <td>0</td>
      <td>98200</td>
      <td>0</td>
      <td>['Fernando Pessoa']</td>
      <td>['14jtPCOoNZwquk5wd9DxrY']</td>
      <td>0.695</td>
      <td>0.2630</td>
      <td>0</td>
      <td>-22.136</td>
      <td>1</td>
      <td>0.9570</td>
      <td>0.797</td>
      <td>0.0000</td>
      <td>0.148</td>
      <td>0.655</td>
      <td>102.009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1922-03-21</th>
      <td>07A5yehtSnoedViJAZkNnc</td>
      <td>Vivo para Quererte - Remasterizado</td>
      <td>0</td>
      <td>181640</td>
      <td>0</td>
      <td>['Ignacio Corsini']</td>
      <td>['5LiOoJbxVSAMkBS2fUm3X2']</td>
      <td>0.434</td>
      <td>0.1770</td>
      <td>1</td>
      <td>-21.180</td>
      <td>1</td>
      <td>0.0512</td>
      <td>0.994</td>
      <td>0.0218</td>
      <td>0.212</td>
      <td>0.457</td>
      <td>130.418</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1922-03-21</th>
      <td>08FmqUhxtyLTn6pAh6bk45</td>
      <td>El Prisionero - Remasterizado</td>
      <td>0</td>
      <td>176907</td>
      <td>0</td>
      <td>['Ignacio Corsini']</td>
      <td>['5LiOoJbxVSAMkBS2fUm3X2']</td>
      <td>0.321</td>
      <td>0.0946</td>
      <td>7</td>
      <td>-27.961</td>
      <td>1</td>
      <td>0.0504</td>
      <td>0.995</td>
      <td>0.9180</td>
      <td>0.104</td>
      <td>0.397</td>
      <td>169.980</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1922-01-01</th>
      <td>08y9GfoqCWfOGsKdwojr5e</td>
      <td>Lady of the Evening</td>
      <td>0</td>
      <td>163080</td>
      <td>0</td>
      <td>['Dick Haymes']</td>
      <td>['3BiJGZsyX9sJchTqcSA7Su']</td>
      <td>0.402</td>
      <td>0.1580</td>
      <td>3</td>
      <td>-16.900</td>
      <td>0</td>
      <td>0.0390</td>
      <td>0.989</td>
      <td>0.1300</td>
      <td>0.311</td>
      <td>0.196</td>
      <td>103.220</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Convert song duration from miliseconds to seconds
df_tracks["duration"] = df_tracks["duration_ms"].apply(lambda x: round(x/1000))
df_tracks.drop("duration_ms", inplace=True, axis=1)
```

```python
df_tracks.duration.head()
```

    release_date
    1922-02-22    127
    1922-06-01     98
    1922-03-21    182
    1922-03-21    177
    1922-01-01    163
    Name: duration, dtype: int64

```python
# Finding Correlation between different variables using Heat Map

corr_df = df_tracks.drop(["key", "mode", "explicit"], axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap = sns.heatmap(corr_df, annot=True, fmt=".1g", vmin=-1, vmax=1, center=0, cmap="inferno", linewidth=1, linecolor="Black")
heatmap.set_title("Correlation HeatMap Between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)
```

    [Text(0.5, 0, 'popularity'),
     Text(1.5, 0, 'danceability'),
     Text(2.5, 0, 'energy'),
     Text(3.5, 0, 'loudness'),
     Text(4.5, 0, 'speechiness'),
     Text(5.5, 0, 'acousticness'),
     Text(6.5, 0, 'instrumentalness'),
     Text(7.5, 0, 'liveness'),
     Text(8.5, 0, 'valence'),
     Text(9.5, 0, 'tempo'),
     Text(10.5, 0, 'time_signature'),
     Text(11.5, 0, 'duration')]

![png](output_10_1.png)

```python
# Taking out 0.4% of data from original data
sample_df = df_tracks.sample(int(0.004*len(df_tracks)))
```

```python
print(len(sample_df))
```

    2346

```python
# Visualizing Correlation between Loudness and Energy

plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y = "loudness", x ="energy", color = "c").set(title = "Loudness vs Energy Correlation" )
```

    [Text(0.5, 1.0, 'Loudness vs Energy Correlation')]

![png](output_13_1.png)

```python
# Visualizing Correlation between Popularity and Acousticness

plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y = "popularity", x ="acousticness", color = "g").set(title = "Popularity vs Acousticness Correlation" )
```

    [Text(0.5, 1.0, 'Popularity vs Acousticness Correlation')]

![png](output_14_1.png)

```python
df_tracks['dates']=df_tracks.index.get_level_values('release_date')
df_tracks.dates=pd.to_datetime(df_tracks.dates)
years = df_tracks.dates.dt.year
```

```python
# Visualizing number of songs released per year
sns.displot(years,discrete=True,aspect=2,height=5, kind="hist").set(title="Number of songs")
```

    <seaborn.axisgrid.FacetGrid at 0x1cd6c936f70>

![png](output_16_1.png)

```python
# Visualzing relation between year and length of songs

total_dr = df_tracks.duration
fig_dims = (18,7)
fig, ax = plt.subplots(figsize = fig_dims)
fig = sns.barplot(x = years, y = total_dr, ax = ax, errwidth = False).set(title="Year vs Duration")
plt.xticks(rotation=90)
```

    (array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
             13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
             26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
             39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
             52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
             65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
             78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
             91,  92,  93,  94,  95,  96,  97,  98,  99, 100]),
     [Text(0, 0, '1900'),
      Text(1, 0, '1922'),
      Text(2, 0, '1923'),
      Text(3, 0, '1924'),
      Text(4, 0, '1925'),
      Text(5, 0, '1926'),
      Text(6, 0, '1927'),
      Text(7, 0, '1928'),
      Text(8, 0, '1929'),
      Text(9, 0, '1930'),
      Text(10, 0, '1931'),
      Text(11, 0, '1932'),
      Text(12, 0, '1933'),
      Text(13, 0, '1934'),
      Text(14, 0, '1935'),
      Text(15, 0, '1936'),
      Text(16, 0, '1937'),
      Text(17, 0, '1938'),
      Text(18, 0, '1939'),
      Text(19, 0, '1940'),
      Text(20, 0, '1941'),
      Text(21, 0, '1942'),
      Text(22, 0, '1943'),
      Text(23, 0, '1944'),
      Text(24, 0, '1945'),
      Text(25, 0, '1946'),
      Text(26, 0, '1947'),
      Text(27, 0, '1948'),
      Text(28, 0, '1949'),
      Text(29, 0, '1950'),
      Text(30, 0, '1951'),
      Text(31, 0, '1952'),
      Text(32, 0, '1953'),
      Text(33, 0, '1954'),
      Text(34, 0, '1955'),
      Text(35, 0, '1956'),
      Text(36, 0, '1957'),
      Text(37, 0, '1958'),
      Text(38, 0, '1959'),
      Text(39, 0, '1960'),
      Text(40, 0, '1961'),
      Text(41, 0, '1962'),
      Text(42, 0, '1963'),
      Text(43, 0, '1964'),
      Text(44, 0, '1965'),
      Text(45, 0, '1966'),
      Text(46, 0, '1967'),
      Text(47, 0, '1968'),
      Text(48, 0, '1969'),
      Text(49, 0, '1970'),
      Text(50, 0, '1971'),
      Text(51, 0, '1972'),
      Text(52, 0, '1973'),
      Text(53, 0, '1974'),
      Text(54, 0, '1975'),
      Text(55, 0, '1976'),
      Text(56, 0, '1977'),
      Text(57, 0, '1978'),
      Text(58, 0, '1979'),
      Text(59, 0, '1980'),
      Text(60, 0, '1981'),
      Text(61, 0, '1982'),
      Text(62, 0, '1983'),
      Text(63, 0, '1984'),
      Text(64, 0, '1985'),
      Text(65, 0, '1986'),
      Text(66, 0, '1987'),
      Text(67, 0, '1988'),
      Text(68, 0, '1989'),
      Text(69, 0, '1990'),
      Text(70, 0, '1991'),
      Text(71, 0, '1992'),
      Text(72, 0, '1993'),
      Text(73, 0, '1994'),
      Text(74, 0, '1995'),
      Text(75, 0, '1996'),
      Text(76, 0, '1997'),
      Text(77, 0, '1998'),
      Text(78, 0, '1999'),
      Text(79, 0, '2000'),
      Text(80, 0, '2001'),
      Text(81, 0, '2002'),
      Text(82, 0, '2003'),
      Text(83, 0, '2004'),
      Text(84, 0, '2005'),
      Text(85, 0, '2006'),
      Text(86, 0, '2007'),
      Text(87, 0, '2008'),
      Text(88, 0, '2009'),
      Text(89, 0, '2010'),
      Text(90, 0, '2011'),
      Text(91, 0, '2012'),
      Text(92, 0, '2013'),
      Text(93, 0, '2014'),
      Text(94, 0, '2015'),
      Text(95, 0, '2016'),
      Text(96, 0, '2017'),
      Text(97, 0, '2018'),
      Text(98, 0, '2019'),
      Text(99, 0, '2020'),
      Text(100, 0, '2021')])

![png](output_17_1.png)

```python
df_genre = pd.read_csv("C:/Users/risha/OneDrive/Documents/Data Analytics and Data Engineering Projects/Spotify Data Analysis Using Python/Datasets/SpotifyFeatures.csv")
```

```python
df_genre.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>artist_name</th>
      <th>track_name</th>
      <th>track_id</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>C'est beau de faire un Show</td>
      <td>0BRjO6ga9RKCKjfDqeFgWV</td>
      <td>0</td>
      <td>0.611</td>
      <td>0.389</td>
      <td>99373</td>
      <td>0.910</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.3460</td>
      <td>-1.828</td>
      <td>Major</td>
      <td>0.0525</td>
      <td>166.969</td>
      <td>4/4</td>
      <td>0.814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Movie</td>
      <td>Martin &amp; les fées</td>
      <td>Perdu d'avance (par Gad Elmaleh)</td>
      <td>0BjC1NfoEOOusryehmNudP</td>
      <td>1</td>
      <td>0.246</td>
      <td>0.590</td>
      <td>137373</td>
      <td>0.737</td>
      <td>0.000</td>
      <td>F#</td>
      <td>0.1510</td>
      <td>-5.559</td>
      <td>Minor</td>
      <td>0.0868</td>
      <td>174.003</td>
      <td>4/4</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Movie</td>
      <td>Joseph Williams</td>
      <td>Don't Let Me Be Lonely Tonight</td>
      <td>0CoSDzoNIKCRs124s9uTVy</td>
      <td>3</td>
      <td>0.952</td>
      <td>0.663</td>
      <td>170267</td>
      <td>0.131</td>
      <td>0.000</td>
      <td>C</td>
      <td>0.1030</td>
      <td>-13.879</td>
      <td>Minor</td>
      <td>0.0362</td>
      <td>99.488</td>
      <td>5/4</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>Dis-moi Monsieur Gordon Cooper</td>
      <td>0Gc6TVm52BwZD07Ki6tIvf</td>
      <td>0</td>
      <td>0.703</td>
      <td>0.240</td>
      <td>152427</td>
      <td>0.326</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.0985</td>
      <td>-12.178</td>
      <td>Major</td>
      <td>0.0395</td>
      <td>171.758</td>
      <td>4/4</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Movie</td>
      <td>Fabien Nataf</td>
      <td>Ouverture</td>
      <td>0IuslXpMROHdEPvSl1fTQK</td>
      <td>4</td>
      <td>0.950</td>
      <td>0.331</td>
      <td>82625</td>
      <td>0.225</td>
      <td>0.123</td>
      <td>F</td>
      <td>0.2020</td>
      <td>-21.150</td>
      <td>Major</td>
      <td>0.0456</td>
      <td>140.576</td>
      <td>4/4</td>
      <td>0.390</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Duration of the Songs in Different Genres

sns.color_palette("rocket", as_cmap = True)
fig_dims = (18,7)
fig= plt.subplots(figsize = fig_dims)
fig = sns.barplot(y='genre', x='duration_ms', data=df_genre).set(title="Duration of the Songs in Different Genres")
plt.xlabel("Duration in milliseconds")
plt.ylabel("Genres ")
```

    Text(0, 0.5, 'Genres ')

![png](output_20_1.png)

```python
# Getting Most Popular Genres

sns.set_style(style="darkgrid")
plt.figure(figsize=(10,5))
famous = df_genre.sort_values("popularity", ascending = False).head(10)
sns.barplot(y="genre", x = "popularity", data = famous).set(title="Top 5 Genres by Popularity")
```

    [Text(0.5, 1.0, 'Top 5 Genres by Popularity')]

![png](output_21_1.png)

```python

```
