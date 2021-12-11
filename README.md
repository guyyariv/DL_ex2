# DL_ex2
 Introduction to Deep Learning - Exercise #2



Programing Task: Sentiment Analysis for the IMDB Movies Reviews Dataset

The dataset consists of 50,000 annotated reviews. Each review is a plain text describing the viewers’ experience and opinion on the movie watched. The dataset consists of highly polar reviews and contains binary labels (positive and negative) based on the number of stars the movie received by the viewers. Some reviews are very long, but we will consider only the first 100 words they contain.

In this exercise we design networks that predict the viewers’ sentiment (positive/negative) towards the movies they watched based on the review they wrote. We will compare the following four different strategies:
The use of a simple Elman RNN
The use of a simple GRU
The use of a global average pooling. In this case every word goes through an MLP that result in a scalar, sub-prediction score, and these scores are then summed together to provide the final prediction
Adding a local self-attention layer to Strategy #3 in order to achieve crossword reasoning

The exercise comes with a partial code that loads the reviews from the dataset, and processes them into a long list of lower-case text (100 words, no punctuation/sentences, no special characters). Longer reviews are truncated, shorter ones are padded. This pre-processing uses the GloVe word embedding which maps every word into a 100-dimensional vector. To avoid over-fitting over this small dataset we do not allow this embedding to train. You can read more about this embedding here. Some parts of these pre-processing steps were taken from this tutorial.

Specific tasks:

Fill in the missing lines of code in the RNN and GRU cells functions. The RNN contains some lines which you may find helpful (or choose to omit and implement on your own). The gates and update operators should consist of a single FC layer (hidden state dim. should be between 64-128 for the lowest test error). The convention of the tensors for these recurrent networks is: batch element x “time” x feature vector. So the recurrence (your iteration in the code) should apply on the second axis. Once the review is parsed, its hidden-state should pass through an MLP  which produces the final output sentiment prediction (a 2-class one hot vector).
Run each of these two recurrent network architectures,describe your experiments with the hidden-state dimension and the train/test accuracies obtained.
In a second experiment you will process each word by an MLP to obtain the 2-class predictions per each word, which we call  “sub prediction scores”, and then sum up all these vectors to obtain a final prediction (which still allows us to handle data of variable length). 
Describe your  experiments with the number of FCs layers and their inner dim and report the best performing one you found. 
Since this model gives a score per each word, output two test examples with these numbers next to their corresponding words (reviews_text and sub_score in the code). The examples should be selected such that one which the prediction is right and one which is wrong. Come up with an explanation of why this happened. To conduct this experiment you are welcome to write your own test reviews in loader.py in the my_test_texts list. 

Write a restricted self-attention layer which queries every word with its closest 5 words on each side (using torch.roll and padding of size 5). This layer should have a single head and learnable query, key and value matrices. You can use the incomplete code in ExLRestSelfAtten to implement this layer or use your own code. This code can use torch.roll to shift the text right and left as well as torch.pad to handle the boundaries.

Finally, add this layer to the network architecture you used in Task 3 and repeat the experiments above; two reviews, one correctly predicted, one wrong, print the sub prediction scores per each word, and explain the results - how they differ from before. Include both these printouts in your report as well as describe the main principle difference in the predictions abilities that this layer adds to the network and how it can be seen in the results.

We are expecting you to report and elaborate on every practical task in the pdf, with your own words and analysis of what you’ve done. Include everything that you think is crucial for us to understand your way of thinking.




