This is a colab project that uses transfer learning using bert model with the help of bert-serving-client service on google
colab to perform sentence embedding for NLP tasks.

Here I have used the IMDB movie review dataset for sentiment classification which can be found in the repo.

For using bert-serving-client as a service please keep in mind the following things:

- Connect to a GPU runtime.
- change the GPU version to 1.15 before using bert-serving-client as it is not supported by the latest tensorflow versions.

