# -*- coding: utf-8 -*-


import pandas as pd
import os
import tempfile
from typing import Dict, Texte
import numpy as np
import tensorflow as tf
from collections import defaultdict

!pip install tensorflow_recommenders

import tensorflow_recommenders as tfrs

from google.colab import drive
drive.mount('/content/drive')


tdata = pd.read_csv('/content/drive/MyDrive/Assignment 2/test_0.txt', sep="\t", header=None)
tdata = tdata.rename(columns={0:'user_id', 1:'song_id', 2:'rating'})
trdata = pd.read_csv('/content/drive/MyDrive/Assignment 2/train_0.txt', sep="\t", header=None)
trdata = tdata.rename(columns={0:'user_id', 1:'song_id', 2:'rating'})


tr_df = trdata
t_df = tdata

songs = pd.read_csv('/content/drive/MyDrive/Assignment 2/song-attributes.txt', sep="\t", header=None)
songs = songs.rename(columns={0:'song_id', 1:'album_id', 2:'artist_id', 3:'genre_id'})

genres = pd.read_csv('/content/drive/MyDrive/Assignment 2/genre-hierarchy.txt', sep="\t", header=None)
genres = genres.rename(columns={0:'genre_id', 1:'parent_genre', 2:'level', 3:'genre_name'})

df = tr_df.merge(songs, on='song_id').merge(genres, on='genre_id')


"""## Baseline Model"""

t_df = t_df.sample(frac=1)
ratingsTrain = t_df.values[:1_800_000]
ratingsTest = t_df.values[1_800_000:]

ratingsPerUser = defaultdict(list)
ratingsPerSong = defaultdict(list)
for i in ratingsTrain:
  u = i[0]
  s = i[1]
  r = i[2]
  ratingsPerUser[u].append((s, r))
  ratingsPerSong[s].append((u, r))

trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)
globalAverage

validMSE = 0
for i in ratingsTrain:
  u = i[0]
  s = i[1]
  r = i[2]
  se = (r - globalAverage)**2
  validMSE += se

validMSE /= len(ratingsTrain)

print("Validation MSE (average only) = " + str(validMSE))

betaU = {}
betaI = {}
for u in ratingsPerUser:
    betaU[u] = 0

for s in ratingsPerSong:
    betaI[s] = 0
alpha = globalAverage

def iterate(lamb):
    newAlpha = 0
    for u, s, r in ratingsTrain:
        newAlpha += r - (betaU[u] + betaI[s])
    alpha = newAlpha / len(ratingsTrain)
    for u in ratingsPerUser:
        newBetaU = 0
        for s,r in ratingsPerUser[u]:
            newBetaU += r - (alpha + betaI[s])
        betaU[u] = newBetaU / (lamb + len(ratingsPerUser[u]))
    for s in ratingsPerSong:
        newBetaI = 0
        for u,r in ratingsPerSong[s]:
            newBetaI += r - (alpha + betaU[u])
        betaI[s] = newBetaI / (lamb + len(ratingsPerSong[s]))
    mse = 0
    for u,s,r in ratingsTrain:
        prediction = alpha + betaU[u] + betaI[s]
        mse += (r - prediction)**2
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for s in betaI:
        regularizer += betaI[s]**2
    mse /= len(ratingsTrain)
    return mse, mse + lamb*regularizer

mse,objective = iterate(1)
newMSE,newObjective = iterate(1)
iterations = 2

iterations = 1
while iterations < 10 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(5)
    iterations += 1
    print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after " + str(iterations) + " iterations = " + str(newMSE))

validMSE = 0
predictions = []
for u,b,r in ratingsTest:
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    prediction = alpha + bu + bi
    predictions.append(prediction)
    validMSE += (r - prediction)**2

validMSE /= len(ratingsTest)
print("Validation MSE = " + str(validMSE))

m1pred = predictions

"""##Model 2"""

#Randomly shuffle train data and convert values to str
tr_df = tr_df.sample(frac=1)
tr_df['user_id'] = tr_df['user_id'].astype(str)
tr_df['song_id'] = tr_df['song_id'].astype(str)
tr_df['rating'] = tr_df['rating'].astype(str)
#Randomly shuffle test data and convert values to str
t_df = t_df.sample(frac=1)
t_df['user_id'] = t_df['user_id'].astype(str)
t_df['song_id'] = t_df['song_id'].astype(str)
t_df['rating'] = t_df['rating'].astype(str)

#Convert train and test data to tensors
tr_data = tf.convert_to_tensor(np.array(tr_df[['user_id', 'song_id', 'rating']]))
t_data = tf.convert_to_tensor(np.array(t_df[['user_id', 'song_id', 'rating']]))

#convert_to_tensor only takes values with one dtype, re-convert ratings values to ints and add all values as dict entries
tr_dataset = tf.data.Dataset.from_tensor_slices(tr_data)
tr_dataset = tr_dataset.map(lambda x: {
    "song_id": x[1],
    "user_id": x[0],
    "user_rating": int(x[2])
})
t_dataset = tf.data.Dataset.from_tensor_slices(t_data)
t_dataset = t_dataset.map(lambda x: {
    "song_id": x[1],
    "user_id": x[0],
    "user_rating": int(x[2])
})

#seperate user_ids and song_ids for vocabulary generation
user_ids = tr_dataset.batch(1_000_000).map(lambda x: x['user_id'])
song_ids = tr_dataset.batch(1_000_000).map(lambda x: x['song_id'])
#find unique song and user ids to create vocabulary 
unique_song_ids = np.unique(np.concatenate(list(song_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

len(unique_user_ids)

class RankingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        
        self.song_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_song_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_song_ids) + 1, embedding_dimension)
        ])
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, inputs):

        user_id, song_id = inputs

        user_embedding = self.user_embeddings(user_id)
        song_embedding = self.song_embeddings(song_id)

        return self.ratings(tf.concat([user_embedding, song_embedding], axis=1))

class SongModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["user_id"], features["song_id"]))
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model((features['user_id'], features['song_id']))
        return self.task(labels=features['user_rating'], predictions=rating_predictions)

model = SongModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.3))
cached_train = tr_dataset.shuffle(100_000).batch(8192).cache()
cached_test = t_dataset.batch(4096).cache()
model.fit(cached_train, epochs = 6)

model.evaluate(cached_test, return_dict=True)

tf.saved_model.save(model, "export")

!zip -r /content/file.zip "/content/export"

test_ratings = {}
test_song_ids = ['82446']
for song_id in test_song_ids:
    test_ratings[song_id] = model({
        'user_id':np.array(['0']),
        'song_id':np.array([song_id])
    })
print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")

scores = {}
for i in t_df['song_id'].unique():
  scores[i] = np.array(model({
      'user_id':np.array(['18385']),
      'song_id':np.array([i])
  }))[0][0]

sscores = sorted(scores, key = scores.get, reverse=True)
sorted_dict = {}
for i in sscores:
  sorted_dict[i] = scores[i]

relgenres = []
for i in sscores[:1000]:
  relgenres.append(df[df['song_id'] == int(i)].genre_name.iloc[0])

pd.Series([i for i in relgenres if i != 'Unknown']).value_counts()

u = df[df['user_id'] == 18385]
u[u['genre_name'] != "Unknown"]

u = t_df[t_df['user_id'] == '18385']
u

import zipfile
with zipfile.ZipFile("file.zip","r") as zip_ref:
  zip_ref.extractall("export")

loaded = tf.saved_model.load("/content/export/content/export")

loaded

mod1preds = []
for i in ratingsTest:
    song_id = str(i[1])
    user_id = str(i[0])
    mod1preds.append(np.array(loaded({
        'user_id':np.array([user_id]),
        'song_id':np.array([song_id])
    }))[0][0])

cdf = pd.DataFrame()
cdf['song/user'] = [(x[1], x[0]) for x in ratingsTest]
cdf = cdf.set_index('song/user')
cdf['model_1_predictions'] = predictions
cdf['model_2_predictions'] = mod2preds
cdf['actual'] = [x[2] for x in ratingsTest]

cdf.sample(frac=1).head(20)

