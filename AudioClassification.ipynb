{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Import and Install Dependencies"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.1 Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 tensorflow-io matplotlib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.2 Load Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "from matplotlib import pyplot as plt\n",
    "import tensorflow as tf \n",
    "import tensorflow_io as tfio"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Build Data Loading Function"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.1 Define Paths to Files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')\n",
    "NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.2 Build Dataloading Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def load_wav_16k_mono(filename):\n",
    "    # Load encoded wav file\n",
    "    file_contents = tf.io.read_file(filename)\n",
    "    # Decode wav (tensors by channels) \n",
    "    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)\n",
    "    # Removes trailing axis\n",
    "    wav = tf.squeeze(wav, axis=-1)\n",
    "    sample_rate = tf.cast(sample_rate, dtype=tf.int64)\n",
    "    # Goes from 44100Hz to 16000hz - amplitude of the audio signal\n",
    "    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)\n",
    "    return wav"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.3 Plot Wave"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "wave = load_wav_16k_mono(CAPUCHIN_FILE)\n",
    "nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "plt.plot(wave)\n",
    "plt.plot(nwave)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Create Tensorflow Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.1 Define Paths to Positive and Negative Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')\n",
    "NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.2 Create Tensorflow Datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "pos = tf.data.Dataset.list_files(POS+'\\*.wav')\n",
    "neg = tf.data.Dataset.list_files(NEG+'\\*.wav')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.3 Add labels and Combine Positive and Negative Samples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))\n",
    "negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))\n",
    "data = positives.concatenate(negatives)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Determine Average Length of a Capuchin Call"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.1 Calculate Wave Cycle Length"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "lengths = []\n",
    "for file in os.listdir(os.path.join('data', 'Parsed_Capuchinbird_Clips')):\n",
    "    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Parsed_Capuchinbird_Clips', file))\n",
    "    lengths.append(len(tensor_wave))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.2 Calculate Mean, Min and Max"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "tf.math.reduce_mean(lengths)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "tf.math.reduce_min(lengths)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "tf.math.reduce_max(lengths)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Build Preprocessing Function to Convert to Spectrogram"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5.1 Build Preprocessing Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def preprocess(file_path, label): \n",
    "    wav = load_wav_16k_mono(file_path)\n",
    "    wav = wav[:48000]\n",
    "    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)\n",
    "    wav = tf.concat([zero_padding, wav],0)\n",
    "    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)\n",
    "    spectrogram = tf.abs(spectrogram)\n",
    "    spectrogram = tf.expand_dims(spectrogram, axis=2)\n",
    "    return spectrogram, label"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5.2 Test Out the Function and Viz the Spectrogram"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "spectrogram, label = preprocess(filepath, label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(30,20))\n",
    "plt.imshow(tf.transpose(spectrogram)[0])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Create Training and Testing Partitions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6.1 Create a Tensorflow Data Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = data.map(preprocess)\n",
    "data = data.cache()\n",
    "data = data.shuffle(buffer_size=1000)\n",
    "data = data.batch(16)\n",
    "data = data.prefetch(8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6.2 Split into Training and Testing Partitions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train = data.take(36)\n",
    "test = data.skip(36).take(15)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6.3 Test One Batch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "samples, labels = train.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "samples.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Build Deep Learning Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7.1 Load Tensorflow Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, Dense, Flatten"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7.2 Build Sequential Model, Compile and View Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))\n",
    "model.add(Conv2D(16, (3,3), activation='relu'))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(128, activation='relu'))\n",
    "model.add(Dense(1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7.3 Fit Model, View Loss and KPI Plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "hist = model.fit(train, epochs=4, validation_data=test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "plt.title('Loss')\n",
    "plt.plot(hist.history['loss'], 'r')\n",
    "plt.plot(hist.history['val_loss'], 'b')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "plt.title('Precision')\n",
    "plt.plot(hist.history['precision'], 'r')\n",
    "plt.plot(hist.history['val_precision'], 'b')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "plt.title('Recall')\n",
    "plt.plot(hist.history['recall'], 'r')\n",
    "plt.plot(hist.history['val_recall'], 'b')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8. Make a Prediction on a Single Clip"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8.1 Get One Batch and Make a Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_test, y_test = test.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "yhat = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8.2 Convert Logits to Classes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "# 9. Build Forest Parsing Functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9.1 Load up MP3s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "def load_mp3_16k_mono(filename):\n",
    "    \"\"\" Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. \"\"\"\n",
    "    res = tfio.audio.AudioIOTensor(filename)\n",
    "    # Convert to tensor and combine channels \n",
    "    tensor = res.to_tensor()\n",
    "    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 \n",
    "    # Extract sample rate and cast\n",
    "    sample_rate = res.rate\n",
    "    sample_rate = tf.cast(sample_rate, dtype=tf.int64)\n",
    "    # Resample to 16 kHz\n",
    "    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)\n",
    "    return wav"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "mp3 = os.path.join('data', 'Forest Recordings', 'recording_00.mp3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "wav = load_mp3_16k_mono(mp3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "samples, index = audio_slices.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9.2 Build Function to Convert Clips into Windowed Spectrograms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "def preprocess_mp3(sample, index):\n",
    "    sample = sample[0]\n",
    "    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)\n",
    "    wav = tf.concat([zero_padding, sample],0)\n",
    "    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)\n",
    "    spectrogram = tf.abs(spectrogram)\n",
    "    spectrogram = tf.expand_dims(spectrogram, axis=2)\n",
    "    return spectrogram"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9.3 Convert Longer Clips into Windows and Make Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)\n",
    "audio_slices = audio_slices.map(preprocess_mp3)\n",
    "audio_slices = audio_slices.batch(64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "yhat = model.predict(audio_slices)\n",
    "yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9.4 Group Consecutive Detections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "from itertools import groupby"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "yhat = [key for key, group in groupby(yhat)]\n",
    "calls = tf.math.reduce_sum(yhat).numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "calls"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 10. Make Predictions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10.1 Loop over all recordings and make predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "results = {}\n",
    "for file in os.listdir(os.path.join('data', 'Forest Recordings')):\n",
    "    FILEPATH = os.path.join('data','Forest Recordings', file)\n",
    "    \n",
    "    wav = load_mp3_16k_mono(FILEPATH)\n",
    "    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)\n",
    "    audio_slices = audio_slices.map(preprocess_mp3)\n",
    "    audio_slices = audio_slices.batch(64)\n",
    "    \n",
    "    yhat = model.predict(audio_slices)\n",
    "    \n",
    "    results[file] = yhat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10.2 Convert Predictions into Classes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "class_preds = {}\n",
    "for file, logits in results.items():\n",
    "    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]\n",
    "class_preds"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10.3 Group Consecutive Detections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "postprocessed = {}\n",
    "for file, scores in class_preds.items():\n",
    "    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()\n",
    "postprocessed"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Export Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "import csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "with open('results.csv', 'w', newline='') as f:\n",
    "    writer = csv.writer(f, delimiter=',')\n",
    "    writer.writerow(['recording', 'capuchin_calls'])\n",
    "    for key, value in postprocessed.items():\n",
    "        writer.writerow([key, value])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "audioc",
   "language": "python",
   "name": "audioc"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
