# Glossary

Use this page to record terms and ideas that help you understand
professional analytics projects.

This project covers applied machine learning: calling deployed models,
investigating their behavior, and working with text and image data.

Pro-tip: Expand the VS Code **Outline** view (below the navigator on the right)
to see this file organization at-a-glance.

## Deployed Models and APIs

### API (Application Programming Interface)

An API is a defined way for one program to send requests to another.
A prediction API accepts input data and returns a model's prediction as a response.

### endpoint

An endpoint is a specific URL that an API exposes for a particular action.
The `/predict` endpoint in this project receives feature values and returns a predicted species.

### payload

A payload is the data sent in a request to an API.
In this project, the payload is a JSON object containing the feature values for one penguin.

### HTTP POST request

An HTTP POST request sends data to a server and receives a response.
The `requests.post()` function in Python sends a payload to an API endpoint
and returns the server's response, including the prediction.

### cold start

A cold start happens when a server that has been sleeping wakes up to handle a request.
Free hosting tiers often sleep after inactivity, causing the first request to be slow.

### edge case

An edge case is an input that is unusual, extreme, or outside the expected range.
Testing edge cases reveals whether an API handles bad input gracefully or crashes.

### feature sensitivity

Feature sensitivity describes how much a model's prediction changes
when one input feature is varied while others are held fixed.
A feature with high sensitivity has strong influence over the model's output.

## Text as Data

### corpus

A corpus is a collection of text documents used for analysis or training.
Each document in a corpus may be a sentence, paragraph, article, or other text unit.

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a method for representing text as numbers.
Each word in a document gets a weight that is high when the word appears
frequently in that document but rarely across the full corpus.
TF-IDF captures which words appear but discards word order and meaning.

### vocabulary

The vocabulary is the set of unique words the TF-IDF vectorizer learns from the training corpus.
New documents are represented using only the words already in the vocabulary.

### word order

Word order is the sequence in which words appear in a sentence.
TF-IDF ignores word order, so "dog bites man" and "man bites dog" produce identical vectors.
This is one of TF-IDF's known limitations.

### n-gram

An n-gram is a sequence of n consecutive words treated as a single feature.
Using bigrams (n=2) gives TF-IDF some ability to capture word pairs,
partially recovering word order information.

## Images as Data

### pixel

A pixel is one point of color or brightness in an image.
A grayscale 8x8 image contains 64 pixels, each with a value from 0 (black) to 16 (white).

### flattening

Flattening converts a 2D image grid into a 1D array of pixel values.
An 8x8 image becomes a vector of 64 numbers that a classifier can use as features.
Flattening discards spatial structure - which pixels are neighbors - which limits
a model's ability to handle rotated or shifted images.

### spatial structure

Spatial structure refers to the arrangement and relationships between pixels in an image.
Flattening an image into a feature vector removes spatial structure.
More advanced representations such as convolutional neural networks preserve it.

### confusion matrix

A confusion matrix shows how often a classifier predicted each class
compared to the true class.
Off-diagonal entries are misclassifications.
For digit images, a confusion matrix reveals which digit pairs the model confuses most often.
