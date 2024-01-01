# Install the transformers library if not already installed
!pip install transformers

# Import necessary libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import tensorflow as tf
import numpy as np
import torch
from torch import nn

# Create a class called GrammarCorrection
class GrammarCorrection:

    def __init__(self):
        # Initialize the model and the tokenizer.
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Define a method to detect anomalies in original tokens
    def check(self, dataset, best_probability, checklist):
        result = []
        final_sentences = []
        checklist=self.tokenizer.encode(checklist[0], return_tensors="pt", add_special_tokens=True)

        # Initialize the Softmax function
        m = nn.Softmax(dim=1)

        # Loop through each sentence in the dataset
        for index, eval_sentence in enumerate(dataset):

            # Tokenize the input sentence and convert it to PyTorch tensor
            inputs = self.tokenizer(eval_sentence, return_tensors="pt")
            tokens_tensor = self.tokenizer.encode(eval_sentence, add_special_tokens=False, max_length=1020, truncation=True)
            sentence = []  # Initialize a list to store the tokens of the sentence after checking grammar
            sentence.append(tokens_tensor[0]) # append the first word in the sentence

            # Generate model outputs
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            theirprobabilities = []

            # Loop through tokens in the sentence
            for i in range(1, len(tokens_tensor)):
                pre_word = tokens_tensor[i - 1]
                curr_word = tokens_tensor[i]

                # Calculate word probabilities using Softmax
                softmax_output = m(outputs[1][:, i - 1, :])
                word_probability = softmax_output[0][curr_word]
                word_probability = word_probability.detach().numpy()

                # Check if word probability is below a threshold
                if word_probability < best_probability:
                    if curr_word in checklist:
                        probabilities = []

                        # Calculate probabilities for specific words in theirList
                        for j in range(len(checklist.tolist()[0])):
                            p = softmax_output[0][checklist.tolist()[0][j]]
                            p = p.detach().numpy()
                            probabilities.append(p)
                        print(probabilities)
                        # Find the index of the highest probability
                        max_prob_index = np.argmax(probabilities)

                        # Check if the highest probability is different from the current word's index
                        if max_prob_index != checklist.tolist()[0].index(curr_word):
                            sentence.append(checklist.tolist()[0][max_prob_index])
                        else:
                            sentence.append(curr_word)
                    else:
                        sentence.append(curr_word)
                else:
                    sentence.append(curr_word)

            print(self.tokenizer.decode(sentence))  # Decode the sentence
            final_sentences.append(sentence)
            sentence = []  # Reset the sentence list
        return final_sentences

dataset=['This is there books.', 'I have been their.', 'My house their is pretty.', 'I like there cookies.', 'I like their.','There cookies are good.']
theirList=["There there There their Their"]
model = GrammarCorrection()
model.check(dataset, 0.005, theirList)

