# Multilayered (or singlelayered) GRU based RNN for text generation using Keras libraries
# Tested in Tensorflow backend library
# Code by: Jishnu Ray Chowdhury
# License: BSD

# import libraries
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import random
import sys
import os

# load Dataset
"""path = "Witt2.txt" # Extract from Wittgenstein's 'Philosophical Investigations'"""

path = raw_input("Enter file name (example: Wittgenstein.txt) for training and testing data (make sure it's in the same directory):\n ")
dataset = open(path).read().lower()

# store the list of all unique characters in dataset
chars = sorted(list(set(dataset)))

total_chars = len(dataset)
vocabulary = len(chars)

print("Total Characters: ", total_chars)
print("Vocabulary: ", vocabulary)

# Creating dictionary or map in order to map all characters to an integer and vice versa
char_to_int = { c:i for i, c in enumerate(chars)}
int_to_char = { i:c for i, c in enumerate(chars)}

mapvar = raw_input("Do you want to see the character to integer map? (y/n): ")

if mapvar == "y" or mapvar == "Y":
    # Show the map from Char to Int
    print('\nGenerated Map: ')
    for char in chars:
        print(char,' is mapped to ',char_to_int[char],' and vice versa.')

# Asking the important questions
sample_len = int(raw_input("\nLength of sample text: "))
print("\nChoose:")
print("Enter 0 if you want train the model.")
print("Enter 1 if you want to load saved model and weights.")
print("Enter 2 to resume training using last saved model and weights.")
Answer = int(raw_input('Enter: '))

if Answer == 0:
    
    # Tested with:
    # Number of Hidden Layers: 1
    # Number of Neurons in Hidden Layer 1: 100
    # Time Steps: 30
    # Learning Rate: 0.01
    # Dropout Rate: 0.2
    # Batch Size: 125 or something
    # Do try other combinations
    
    hidden_layers = int(raw_input("\nNumber of Hidden Layers: "))
    neurons = []
    for i in xrange(0,hidden_layers):
        neurons.append(int(raw_input("Number of Neurons in Hidden Layer "+str(i+1)+": ")))
    seq_len = int(raw_input("Time Steps: "))
    learning_rate = float(raw_input("Learning Rate: "))
    dropout_rate = float(raw_input("Dropout Rate: "))
    batch = int(raw_input("Training Batch Size: "))

    # Doing some maths so that the total patterns in future become DIVISIBLE by batch size
    # total no. of patterns need to divisible by batch size because each batch must be of the same size..
    # ...so that the RNN layer can be 'Stateful'
    
    index = int((total_chars-seq_len)/batch)
    index = batch*index
    dataset = dataset[:index+seq_len]
    total_chars=len(dataset)
    
    # prepare input data and output(target) data
    # (X signified Inputs and Y signifies Output(targeted-output in this case)
    dataX = []   
    dataY = []

    for i in range(0,total_chars-seq_len):  # Example of an extract of dataset: Language 
        dataX.append(dataset[i:i+seq_len])  # Example Input Data: Languag
        dataY.append(dataset[i+seq_len])    # Example of corresponding Target Output Data: e

    total_patterns = len(dataX) 
    print("\nTotal Patterns: ", total_patterns)

    # One Hot Encoding...
    X = np.zeros((total_patterns, seq_len, vocabulary), dtype=np.bool)
    Y = np.zeros((total_patterns, vocabulary), dtype=np.bool)

    for pattern in xrange(total_patterns):
        for seq_pos in xrange(seq_len):
            vocab_index = char_to_int[dataX[pattern][seq_pos]]
            X[pattern,seq_pos,vocab_index] = 1
        vocab_index = char_to_int[dataY[pattern]]
        Y[pattern,vocab_index] = 1
    
    # build the model: a multi(or single depending on user input)-layered GRU based RNN
    print('\nBuilding model...')

    model = Sequential()
    if hidden_layers == 1:
        model.add(GRU(neurons[0], batch_input_shape=(batch, seq_len, vocabulary), stateful=True))
    else:
        model.add(GRU(neurons[0], batch_input_shape=(batch, seq_len, vocabulary), stateful=True, return_sequences=True))
    model.add(Dropout(dropout_rate))
    for i in xrange(1,hidden_layers):
        model.add(GRU(neurons[i], stateful=True))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(vocabulary))
    model.add(Activation('softmax'))

    RMSprop_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop_optimizer)
    
    # save model information
    model.save('GRUModel.h5')
    f = open('GRUTimeStep','w+')
    f.write(str(seq_len))
    f.close()
    
else:
    print('\nLoading model...')
    try:
        model = load_model('GRUModel.h5')
    except:
        print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved model to load.")
        print("Solution: May be create and train the model anew ?")
        sys.exit(0)
    try:
        seq_len = int(open('GRUTimeStep').read())
    except:
        print("\nUh Oh! Caught some exceptions! May be you are missing the file having time step information")
        seq_len = int(raw_input("Time Steps (I hope, you remember what it was): "))
        f = open('GRUTimeStep','w+')
        f.write(str(seq_len))
        f.close()

        
# define the checkpoint 
filepath="GRUWeights.h5" # Best weights for sampling will be saved here.
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

# Function for creating a sample text from a random seed (an extract from the dataset).
# The seed acts as the input for the GRU RNN and after feed forwarding through the network it produces the output
# (the output can be considered to be the prediction for the next character)

def sample(seed):
    for i in xrange(sample_len):
             # One hot encoding the input seed
            x = np.zeros((batch, seq_len, vocabulary))
             # There was only one pattern (seed) to enter but for stateful to work batch size should be fixed 
             # and a no. of patterns entered in the model must be same as batch size.
             # So I shaped x input as to have batch size of patterns. But only one pattern in x[0,_,_] will be hot encoded
             # and others will be all 0s because we have only one pattern lenth of infromation from seed.
            for seq_pos in xrange(seq_len):
                vocab_index = char_to_int[seed[seq_pos]]
                x[0,seq_pos,vocab_index] = 1
            # procuring the output (or prediction) from the network
            prediction = model.predict(x,batch_size=batch,verbose=0)
            # Since only the first pattern in the input x had value, only the prediction (prediction[0,_]) 
            # for the first pattern in x (x[0,_,_]) is relevant
            prediction = prediction[0] 
            
            # The prediction is an array of probabilities for each unique characters. 
            # Randomly an integer(mapped to a character) is chosen based on its likelihood 
            # as described in prediction list
            
            RNG_int = np.random.choice(range(vocabulary), p=prediction.ravel())          
            
            # The next character (to generate) is mapped to the randomly chosen integer 
            # Procuring the next character from the dictionary by putting in the chosen integer
            next_char = int_to_char[RNG_int] 
            
            # Display the chosen character
            print(next_char, end="")
            sys.stdout.flush()
            # modifying seed for the next iteration for finding the next character
            seed = seed[1:] + next_char
            
    print()
            

if Answer == 0 or Answer == 2:
    # Train Model and print sample text at each epoch.
    for iteration in range(1,60):
        print()
        print('Iteration: ', iteration)
        print()
        
        # Train model. If you have forgotten: X = input, Y = targeted outputs
        model.fit(X, Y, batch_size=batch, nb_epoch=1, callbacks=[checkpoint])
        model.save('GRUModel.h5') # Saving current model state so that even after terminating the program; training
                                  # can be resumed for last state in the next run.
        print()
        
        # Randomly choosing a sequence from dataset to serve as a seed for sampling
        start_index = random.randint(0, total_chars - seq_len - 1)
        seed = dataset[start_index: start_index + seq_len]
        
        sample(seed)
else:
    # load the network weights
    filename = "GRUWeights.h5"
    try:
        model.load_weights(filename)
    except:
        print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved weights to load.")
        print("Solution: May be create and train the model anew ?")
        sys.exit(0)
    Answer2 = "y"
    while Answer2 == "y" or Answer2 == "Y":
            print("\nGenerating Text:\n")
            # Randomly choosing a sequence from dataset to serve as a seed for sampling
            start_index = random.randint(0, total_chars - seq_len - 1)
            seed = dataset[start_index: start_index + seq_len]
            sample(seed)
            print()
            Answer2 = raw_input("Generate another sample Text? (y/n): ")
