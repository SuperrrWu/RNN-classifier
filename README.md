# RNN-classifier
A classifier that classifies categories of poetry.

Provides two different ways to implement a classifier for Chinese:

· The first was when the splitter processed each line of the poem before setting up a frequency dictionary.

· The second is to separate each Chinese word, not according to the common habit of participle finally build a frequency dictionary.

The second is better because, after the word segmentation of jieba, many of the participles in the new poem are missing from the dictionary, resulting in only punctuation being recognised.

You should pay attention:
please change file path!
