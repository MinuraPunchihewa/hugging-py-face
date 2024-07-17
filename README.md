# Hugging-Py-Face

Copyright © 2024 Minura Punchihewa

Hugging-Py-Face is a powerful Python package that provides seamless integration with the Hugging Face Inference API, allowing you to easily perform inference on your machine learning models hosted on the Hugging Face Model Hub.

One of the key benefits of using the Hugging Face Inference API is that it provides a scalable and efficient way to perform inference on your models, by allowing you to easily deploy and serve your models in the cloud. Additionally, the Inference API provides a simple and standardized API that can be used across different programming languages, making it easy to integrate your models with other services and tools.

With Hugging-Py-Face, you can take advantage of these benefits while also enjoying the simplicity and flexibility of using Python.

It allows you to easily customize your API requests, adjust request parameters, handle authentication and access tokens, and interact with a wide range of machine learning models hosted on the Hugging Face Model Hub.

Overall, Hugging-Py-Face is an awesome tool for any machine learning developer or data scientist who wants to perform efficient and scalable inference on their models, while also enjoying the simplicity and flexibility of using Python. Whether you're working on a personal project or a large-scale enterprise application, Hugging-Py-Face can help you achieve your machine learning goals with ease.

## Installation

### With pip

```
pip install hugging_py_face
```

## Components

- NLP (Natural Language Processing): This component deals with processing and analyzing human language. It includes various techniques such as text classification, text generation, summarization and many more.
- Computer Vision: This component deals with the analysis of visual data from the real world. It includes the image classification and object detection techniques.
- Audio Processing: This component deals with the analysis of audio signals. It includes the audio classification and speech recognition techniques.

## Usage

The library will first need to be configured with a User Access Tokens from the Hugging Face website.

### NLP (Natural Language Processing)

```
from hugging_py_face import NLP

# initialize the NLP class with the user access token
nlp = NLP('hf_...')

# perform text classification
nlp.text_classification("I like you. I love you.")

# perform object detection
nlp.text_generation("The answer to the universe is")
```

The inputs to these methods can also be a list of strings. For example:

```
nlp.text_classification(["I like you. I love you.", "I hate you. I despise you."])
```

Additionally, the fill mask, summarization, text classification and text generation tasks can also be performed on a pandas DataFrame. For example:

```
nlp.text_classification_in_df(df, 'text')
# where df is a pandas DataFrame and 'text' is the column name containing the text
```

### Computer Vision

```
from hugging_py_face import ComputerVision

# initialize the ComputerVision class with the user access token
cp = ComputerVision('hf_...')

# perform image classification
# the image can be a local file or a URL
cp.image_classification("cats.jpg")

# perform object detection
# the image can be a local file or a URL
cp.object_detection("cats.jpg")
```

The inputs to these methods can also be a list of images. For example:

```
cp.image_classification(["cats.jpg", "dogs.jpg"])
```

Additionally, the image classification task can also be performed on a pandas DataFrame. For example:

```
cp.image_classification_in_df(df, 'images')
# where df is a pandas DataFrame and 'images' is the column name containing the image file paths or URLs
```

### Audio Processing

```
from hugging_py_face import AudioProcessing

# initialize the AudioProcessing class with the user access token
ap = AudioProcessing('hf_...')

# perform audio classification
# the audio file can be a local file or a URL
ap.audio_classification("dogs.wav")

# perform speech recognition
# the audio file can be a local file or a URL
ap.speech_recognition("dogs.wav")
```

The inputs to these methods can also be a list of audio files. For example:

```
ap.audio_classification(["dogs.wav", "cats.wav"])
```

Additionally, both of the above tasks can also be performed on a pandas DataFrame. For example:

```
ap.audio_classification_in_df(df, 'audio')
# where df is a pandas DataFrame and 'audio' is the column name containing the audio file paths or URLs
```

# License

This code is licensed under the MIT LICENSE. See LICENSE.txt for details.
