# Character based language model

## Usage

### Requirements
- Python 3
- Pip

### Installation
- Create a virtual enviornment (recommended name is `venv` so that it is git ignored by default)
- Activate the environment
- Run `pip install -r requirements.txt`

## Documentation

### Word embeddings trade-off

One problem that might come with using word embeddings instead of character 
embeddings is that with word embeddings you might encounter out of vocabulary tokens elements quite often, another issue with word embeddings is word ambiguity as the same word could mean different things and we might use the same representation if we use static word embeddings instead of in context word embeddings. However advantages could be for sure capturing better semantic representations than character representations which are much more limited and less sensitive to nuances, also by using context embeddings instead of static embeddings of word we avoid the problem of polysemy for the same vectorization, another advantage is that word embeddings are dense concise representations which are much costlier but also of better quality than one hot encoded like word representations how n-grams or bag of words are.

### Better metric

There are several metrics that were chosen, these were the ones that were used and 
which monitor the performance 

- Perplexity
- F1 score
- BLEU score 
- Levenshtein edit distance
- Word Error Rate
- Character Error Rate

### Better stopping criterion

Early stopper was used in order to stop the training after a set number of 
iterations, the parameters for the early stopper were a patience of 50 (so 50 
iterations at least) and a delta value of 0.1, which means that if after 50 
iterations there is not an improvement of a loss at least 0.1 smaller, then the 
training procedure stops.

Not related to stopping itself, but in order to ensure the stopping will come 
earlier and the model will be able to converge more easily and thus stop earlier, an 
important addition was to use a learning rate scheduler that adapts the learning 
rate in order to make it smaller to prevent abrupt changes when necessary or make it 
larger when the gradient changes are too small.

### Implementation

One of the first problems noticed was the fact that at start it was not possible to 
use more than 13 characters for the initial input prompt, as there was no 
implementation of a padding function.

Also, noticeable was the choice of loss function which was a simple 0-1 function and 
the choice was to instead use a cross entropy loss function which yields more 
granular results which help the gradient updates more than the 0-1 loss function.

Intuitively enough the fully connected layers are not as prone to be succesful as 
other types of architectures for this task, due to this, a different forward method 
was developed which was able to use one out of RNN, LSTM or GRU functions instead of 
fully connected.

Another issue was the heaviside loss function which was equal to 0.1 * sigmoid 
function, the solution was to replace it with the relu function which yielded better 
results.

Besides RNN, LSTM and GRU I also implemented batch norm and dropout as layers, even 
though that was probably not a great idea, as the model was rather underfitting 
in the end (with the cross entropy loss falling somewhere between 1.8-2.2 typically)
and not overfitting (and batch norm / dropout would have been useful just 
for overfitting).

### Findings

GRU, LSTM and RNN work better for the task of next character prediction than the 
simple use of fully connected leyers.

One finding was that the model when asked to generate with a smaller temperature it 
was more repetitive and even replied with the same word over and over for 500 
characters of sequence or it was outputting total gibberish words when the 
temperature was 1.0, whilst the model still might exhibit both of these traits of 
gibberish words or repeating things from the training dataset, these behaviors were 
notably less present in its outputs. 

Another finding was that when do_sample parameter was set to False, was that the 
model would repeteadly output the same word, specifically the word "the" - which is 
probably the most common word which occurs in the dataset, as do_sample=False 
determines topk to be used and with k set to 1, this effectively works like a Greedy 
which is far from being optimal for such a model and with the given data, perhaps 
using topk with a k equal to 5 or 10 instead of k=1 could be usable. Also noticeable 
refering to top_k was that even though the top_k parameter existed, the value was 
still left with a hardcoded 1, instead of the actual variable.


### Further ideas

One intuitive idea for which there was not time for implementation would be to use a 
transformer architecture for this task instead of fully connected or

Secondly, changing the loss function from cross entropy with perplexity could be 
useful, I tried using perplexity, but maybe there was a mistake in how I implement 
perplexity or maybe it's just not a good loss function for this case and is just a 
good performance metric.

Another low-hanging fruit probably to be tried would be to use more data for this 
task in order to allow for better generalization on this task.

Also in terms of activation functions, I experimented with sigmoid, tanh and relu 
(which I chose in the end), but maybe there would be some research as well to be 
done regarding which activation function would work.

Last further idea would be to play a lot with the hyperparameters (the embedding 
size, the hidden dimension, but also the block size) as these ones probably 
influence the most the final performance of the model.

### Conclusion

Many things could of course be done, but unfortunately not much can be done and 
tried in only 3 hours for this task, however I am more than happy to discuss more 
about what I have done and what else I could try in order to improve the performance 
of the langauge model for this task.

Also, due to lack of time, I could not calculate all the results for all the 
suggested hyperparameteres from the main.py file (meaning the embedding size, 
hidden dimension and the learning rate), but I chose the ones which seemed to 
display the best results for the task.

### Refs:
Codebase adapted from minGPT: https://github.com/karpathy/minGPT/


[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)

