# Auto-Text-Generation-Keras-GRU

This is a simple implementation of an auto text generator in [Keras](https://keras.io/) with Tensorflow backend.

I did this mainly as practice while trying to get my hands dirty with machine learning. (Also this was my first work related to machine
learning other than a implementation of basic multilayered neural network in C)

This work is based on the character level model as described on [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

There are other implementations of auto-text-generators in Keras available in net. This one only have some minor differences.
I used multilayered (no. of layers specified during runtime) GRU (Gated Recurrent Unit) for this model along with save, load and checkpoint mechanisms; also with some runtime options for loading a model or starting training from scratch and so on. 

More about GRU, [here](https://arxiv.org/abs/1412.3555).

I made two separate implementations: stateful and stateless.

This is what basically happens when stateful is enabled:

>stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.

([source](https://keras.io/layers/recurrent/))

I trained the model on a section of David Hume's "Treatise on Human Nature" downloaded from Project Gutenberg.

It was hard to play around with models like this in my slow computer. I didn't train the model for much epochs; also
I didn't optimize the hyperparameters too.

This is a sample output (on the stateless version, if I recall correctly):

>to assuration of the object of the essence of the senses, there is any view of them in a naturally the under the founded of the probability, and it is the necessary entersiety of the sensation of the object of the relations, in a matter indisent, who we may only be into can never may be proportion of the really in the existent, is all it. i have us, that is thus, who pects or imagination and can in the pection, which will never particular is the a be and expectly imagin our evil of the to sensible to immedial effect of the one imagination of so angerious, that the most impsention to the imaginately in the man be ever would we passion, that of a same receive to the pection"

I don't remember the exact hyperparameters that I used back then to obtain this exact output; but do experiment with your own hyperparameters and your own text corpus.





