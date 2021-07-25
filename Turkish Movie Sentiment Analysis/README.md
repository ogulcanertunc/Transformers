# <h1 align="center"> Turkish Movie Sentiment Analysis </h1>
In this repo, I fine-tuned a transformer model for the Sensitivity classification problem. 
  
If we look at the basics of sentiment classification, we will realize that multi-class classification is actually a special case in NLP. In this case, there are classes that represent the emotion represented by the text, and we try to guess them. The number of classes usually ranges from 3 to 5, but of course, it is likely to vary, generally, classes are found in positive, negative, and neutral.
# Dataset
The Turkish Movie Sentiment dataset available on Kaggle, prepared by Mustafa Keskin and included in HuggingFace. [huggingface](https://huggingface.co/datasets/turkish_movie_sentiment)  
# Implementations
  - [x] BERT (Fine-Tuning)

# Resuts
- `RoBERTa` - Validation Accuracy: 71.77186799102851

# Requirements

* Python 3.6 and above
* Pytorch, Transformers and all the well known Python ML Libraries
* "TPU enabled setup" / I used Kaggle
* Wandb.ai

# Medium Article
You can find the tutorial on Medium [here]() .
