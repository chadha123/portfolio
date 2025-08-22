'''
Template for the 4th assignment
Student: NAME SURNAME
'''

############################
# Packages
############################
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math
import regex as re
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import random
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
############################
# Classes
############################
# Vocabulary class
class Vocabulary:
    '''
    Class for dealing with our corpus
    '''

    def __init__(self, name, pairs):
        """
        Args:
            name (str): name of the language
            pairs (list): list of pairs of sentences
        """
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4
    def word_to_index(self, word):
        """Returns the index of the word, or the index of <UNK> if the word is not found."""
        return self.word2index.get(word, self.word2index["<UNK>"])

    def add_word(self, word):
        '''
        Add a word to the vocabulary
        :param word: a string
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def add_sentence(self, sentence):
        '''
        Add a sentence to the vocabulary
        :param sentence: list of strings (words)
        '''
        if isinstance(sentence, list):
            for word in sentence:
                self.add_word(word)
        elif isinstance(sentence, str):
            for word in sentence.split(' '):
                self.add_word(word)




def clear_punctuation(s):
    '''
    This function removes all the punctuation from a sentence and insert a blank between any letter and !?.
    :param s: a string
    :return: the "cleaned" string
    '''
    re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Remove all the character that are not letters, puntuation or numbers
    # Insert a blank between any letter and !?. using regex
    s = re.sub(r"([a-zA-Z])([!?.])", r"\1 \2", s)
    return s

# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocabulary, pairs):
        self.vocabulary = vocabulary
        self.pairs = pairs


    def __len__(self):
        return len(self.pairs)

       

    def __getitem__(self, ix):
        pair = self.pairs[ix]
        if len(pair) != 2:
            raise ValueError(f"Expected a tuple of 2 elements, got: {pair}")
        source_sentence, target_sentence = pair
   
        # Convert to tensor
        source_tensor = torch.tensor([vocab.word_to_index(word) for word in source_sentence], dtype=torch.long)
        target_tensor = torch.tensor([vocab.word_to_index(word) for word in target_sentence], dtype=torch.long)


        return source_tensor, target_tensor[:-1],target_tensor[1:]
    
    
    @staticmethod
    def collate_fn(batch):
        source_tensors, target_tensors1,target_tensors2 = zip(*batch)
        source_tensors_padded = pad_sequence(source_tensors, batch_first=True, padding_value=0)
        target_tensors_padded1 = pad_sequence(target_tensors1, batch_first=True, padding_value=0)
        target_tensors_padded2 = pad_sequence(target_tensors2, batch_first=True, padding_value=0)
        
        return source_tensors_padded, target_tensors_padded1,target_tensors_padded2
   
   


class PositionalEncoding(nn.Module):
    '''
    Adapted from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            assert x.size(0) < self.max_len
        except:
            print("The length of the sequence is bigger than the max_len of the positional encoding. Increase the max_len or provide a shorter sequence.")
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, num_heads=8, dropout_p=0.1):
        super(TransformerModel, self).__init__()
        # Embedding layer that will convert input indices into embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer

        # Positional Encoding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout_p)

        # The Transformer model itself
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            batch_first=True  # This argument specifies whether the input tensors are provided as (batch, seq, feature)
        )

        # Linear layer that will project the output of the transformer into the vocabulary space
        self.linear = nn.Linear(d_model, vocab_size)

        # Save the pad token id to use it for mask creation
        self.pad_id = pad_id

        # Save the vocab size for use in the linear layer
        self.vocab_size = vocab_size

        # Initialize weights
        self.init_weights()
    
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

        

      
    def create_padding_mask(self, x, pad_id=0):
        """
        Create a boolean mask for <PAD> tokens in the tensor.
        Args:
        - x (Tensor): The input tensor containing token indices.
        - pad_id (int): The index of the <PAD> token in the vocabulary.

        Returns:
        - mask (Tensor): A boolean tensor where positions with <PAD> are True.
        """
        mask = (x == pad_id)
        return mask
           
      



    def forward(self, src, tgt):
        # S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        # src: (N, S)
        # tgt: (N, T)
        # src_pad_mask: (N, S)
        # tgt_pad_mask: (N, T)
        # mask the future : (N * num_heads, T, T)

        # src: Source sequence tensor (batch, src sequence length)
        # tgt: Target sequence tensor (batch, tgt sequence length)

        # Create padding masks for source and target
        src_pad_mask = self.create_padding_mask(src, self.pad_id)  # (batch, src sequence length)
        tgt_pad_mask = self.create_padding_mask(tgt, self.pad_id)  # (batch, tgt sequence length)

        # Embedding + Positional Encoding for source and target sequences
        src = self.embedding(src)  # (batch, src sequence length, d_model)
        tgt = self.embedding(tgt)  # (batch, tgt sequence length, d_model)

        src = self.pos_encoder(src)  # (batch, src sequence length, d_model)
        tgt = self.pos_encoder(tgt)  # (batch, tgt sequence length, d_model)

        # Future mask for target sequence to prevent attending to subsequent positions
        tgt_seq_length = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(tgt.device)  # (tgt sequence length, tgt sequence length)

        # The Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=tgt_pad_mask,
                                  memory_key_padding_mask=src_pad_mask)  # (batch, tgt sequence length, d_model)

        # Linear layer to project to vocabulary size
        output = self.linear(output)  # (batch, tgt sequence length, vocab_size)

        return output
  

def read_movie_lines(file_path):
    """
    Reads the movie lines from the given file and returns a dictionary
    where keys are line IDs and values are the actual lines.
    """
    lines = {}
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                lines[parts[0]] = parts[4].strip()
    return lines
def read_movie_conversations(file_path):
    """
    Reads the movie conversations from the given file and returns a list
    of conversations where each conversation is a list of line IDs.
    """
    conversations = []
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                conversation = eval(parts[3])
                conversations.append(conversation)
    return conversations


# Create the pairs
def create_sentence_pairs(movie_lines, movie_conversations):
    """
    Creates sentence pairs from the movie conversations.
    """
    pairs = []
    for conversation in movie_conversations:
        for i in range(len(conversation) - 1):
            line1 = movie_lines.get(conversation[i])
            line2 = movie_lines.get(conversation[i+1])
            if line1 and line2:
                pairs.append((line1, line2))
    return pairs


def tokenize_sentence(sentence, is_answer=False):
    """
    Tokenizes the sentence, handling punctuation and special tokens.
    """
    sentence = clear_punctuation(sentence)
    tokens = sentence.split()
    if is_answer:
        tokens = ['<SOS>'] + tokens
    tokens.append('<EOS>')
    return tokens





# Training loop 
def train(model, train_dataloader, val_dataloader, epochs, criterion, optimizer, scheduler, target_loss=1.5):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt1,tgt2 in train_dataloader:
            
            optimizer.zero_grad()
            output = model(src, tgt1)  # Exclude <EOS> for target in input
            loss = criterion(output.transpose(1, 2), tgt2)  # Shift target for loss calculation; exclude <SOS>
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt_input, tgt_output in val_dataloader:
                if src.shape[0] != tgt_input.shape[0]:
                    # Skip this batch if the batch sizes of src and tgt do not match
                    continue
                
                output = model(src, tgt_input)
                loss = criterion(output.transpose(1, 2), tgt_output)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early stopping condition
        if avg_val_loss <= target_loss:
            print(f"Target validation loss achieved at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss by normal training')
    plt.legend()
    plt.show()

    return model
def train_ga(model, train_dataloader, val_dataloader, epochs, criterion, optimizer, scheduler, accumulation_steps=32, target_loss=1.2):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, accum_loss = 0.0, 0.0
        for i, (src, tgt) in enumerate(train_dataloader):
            output = model(src, tgt[:-1])
            loss = criterion(output.transpose(1, 2), tgt[1:]) / accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            if (i + 1) % accumulation_steps == 0 or i + 1 == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accum_loss
                accum_loss = 0.0

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dataloader:
                output = model(src, tgt[:-1])
                loss = criterion(output.transpose(1, 2), tgt[1:])
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early stopping condition
        if avg_val_loss <= target_loss:
            print(f"Target validation loss achieved at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss with Gradient Accumulation')
    plt.legend()
    plt.show()

    return model





# Evaluation 
def generate_greedy(model, input_sentence, vocabulary, max_length=50):
    model.eval()
    with torch.no_grad():
        # Tokenize and numericalize the input sentence
        input_tokens = ['<SOS>'] + tokenize_sentence(input_sentence) + ['<EOS>']
        input_indices = [vocabulary.word2index[token] for token in input_tokens]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)

        # Initialize the target sequence with <SOS>
        output_indices = [vocabulary.word2index['<SOS>']]

        for _ in range(max_length):
            target_tensor = torch.tensor([output_indices], dtype=torch.long)
            
            # Forward pass through the model
            output = model(input_tensor, target_tensor)

            # Select the word with the highest probability
            next_word_idx = output[0, -1, :].argmax().item()
            output_indices.append(next_word_idx)

            # Stop if <EOS> token is generated
            if next_word_idx == vocabulary.word2index['<EOS>']:
                break

        # Convert indices to words
        generated_sentence = [vocabulary.index2word[idx] for idx in output_indices[1:]]  # Skip <SOS>

    return ' '.join(generated_sentence)
def generate_top_k_sampling(model, input_sentence, vocabulary, k=5, max_length=50):
    model.eval()
    with torch.no_grad():
        # Tokenize and numericalize the input sentence
        input_tokens = ['<SOS>'] + tokenize_sentence(input_sentence) + ['<EOS>']
        input_indices = [vocabulary.word2index[token] for token in input_tokens]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)

        # Initialize the target sequence with <SOS>
        output_indices = [vocabulary.word2index['<SOS>']]

        for _ in range(max_length):
            target_tensor = torch.tensor([output_indices], dtype=torch.long)
            
            # Forward pass through the model
            output = model(input_tensor, target_tensor)

            # Get logits and apply softmax
            logits = output[0, -1, :]
            probs = F.softmax(logits, dim=0)

            # Sample from the top k most likely words
            topk_probs, topk_indices = probs.topk(k)
            topk_probs = topk_probs.tolist()
            topk_indices = topk_indices.tolist()

            next_word_idx = random.choices(topk_indices, weights=topk_probs, k=1)[0]
            output_indices.append(next_word_idx)

            # Stop if <EOS> token is generated
            if next_word_idx == vocabulary.word2index['<EOS>']:
                break

        # Convert indices to words
        generated_sentence = [vocabulary.index2word[idx] for idx in output_indices[1:]]  # Skip <SOS>

    return ' '.join(generated_sentence)




def filter_long_sentences(pairs, max_length):
    """
    Filters out pairs where either sentence is longer than max_length.
    """
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0].split()) <= max_length and len(pair[1].split()) <= max_length:
            filtered_pairs.append(pair)
    return filtered_pairs

############################
#bonus questions
############################
from accelerate import Accelerator

accelerator = Accelerator()
def train_ga_hf(model, train_dataloader, val_dataloader, epochs, criterion, optimizer, scheduler, accumulation_steps=32, target_loss=1.2):
    train_losses, val_losses = [], []

    # Prepare the model, optimizer, and dataloaders for distributed and mixed-precision training
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    for epoch in range(epochs):
        model.train()
        total_loss, accum_loss = 0.0, 0.0

        for step, (src, tgt_input, tgt_output) in enumerate(train_dataloader):
            output = model(src, tgt_input)
            loss = criterion(output.transpose(1, 2), tgt_output) / accumulation_steps
            accelerator.backward(loss)
            accum_loss += loss.item()

            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accum_loss
                accum_loss = 0.0

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt_input, tgt_output in val_dataloader:
                output = model(src, tgt_input)
                loss = criterion(output.transpose(1, 2), tgt_output)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early stopping condition
        if avg_val_loss <= target_loss:
            print(f"Target validation loss achieved at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        return model, train_losses, val_losses
    

class TransformerSeparate(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerSeparate, self).__init__()

        # Define the embedding layers for both source and target sequences
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Define the TransformerEncoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=num_encoder_layers
        )

        # Define the TransformerDecoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers=num_decoder_layers
        )

        # Define the output layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # Embed the source and target sequences
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)

        # Apply the encoder to the source sequence
        src_encoded = self.encoder(src_embedded)

        # Apply the decoder to the target sequence with encoder output
        tgt_decoded = self.decoder(tgt_embedded, src_encoded)

        # Linear transformation for output
        output = self.fc(tgt_decoded)

        return output


def beam_search(model, src_sequence, max_length, beam_width):
    
    model.eval()
    with torch.no_grad():
        src_sequence = src_sequence.unsqueeze(0)  # Add batch dimension
        src_mask = model.generate_square_subsequent_mask(src_sequence.size(1)).to(src_sequence.device)

        # Encode the source sequence
        src_encoded = model.encoder(src_sequence, src_mask)

        # Initialize the beam search candidates
        initial_candidates = [([], 0.0)]
        final_candidates = []

        for step in range(max_length):
            candidates = []

            for seq, seq_score in initial_candidates:
                if len(seq) > 0:
                    # If the sequence is not empty, predict the next token
                    tgt_input = torch.tensor([seq[-1]], dtype=torch.long, device=src_sequence.device)
                    tgt_mask = model.generate_square_subsequent_mask(len(seq)).to(src_sequence.device)
                    tgt_output = model.decoder(tgt_input.unsqueeze(0), src_encoded, tgt_mask)
                    tgt_output = tgt_output.squeeze(0)[-1]  # Get the last token's output

                    # Apply log softmax to convert to probabilities
                    log_probs = F.log_softmax(model.fc(tgt_output), dim=-1)

                    # Get the top-k token candidates
                    topk_scores, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        next_token = topk_indices[i].item()
                        next_score = seq_score + topk_scores[i].item()
                        new_seq = seq + [next_token]
                        candidates.append((new_seq, next_score))
                else:
                    # If the sequence is empty, predict the start token
                    tgt_input = torch.tensor([model.start_token], dtype=torch.long, device=src_sequence.device)
                    tgt_output = model.decoder(tgt_input.unsqueeze(0), src_encoded)
                    tgt_output = tgt_output.squeeze(0)[-1]

                    # Apply log softmax to convert to probabilities
                    log_probs = F.log_softmax(model.fc(tgt_output), dim=-1)

                    # Get the top-k token candidates
                    topk_scores, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        next_token = topk_indices[i].item()
                        next_score = topk_scores[i].item()
                        candidates.append(([next_token], next_score))

            # Sort the candidates by their scores and keep the top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            initial_candidates = candidates[:beam_width]

        final_candidates = initial_candidates

        # Sort the final candidates by their scores
        final_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return the top candidate sequences and their scores
        return final_candidates

# Example usage:
# result = beam_search(trained_model, src_sequence, max_length=20, beam_width=5)

   


############################
# Methods
############################

if __name__ == "__main__":
    # !!! Don't change the seed !!!
    torch.manual_seed(42)
    # !!!!!!
    # Download the data
    
    # Paths to data files
    movie_lines_file = '/Users/chadha/Desktop/HAHI/movie_lines.txt'
    movie_conversations_file = '/Users/chadha/Desktop/HAHI/movie_conversations.txt'
    

    # Read and process the files
    movie_lines = read_movie_lines(movie_lines_file)
    movie_conversations = read_movie_conversations(movie_conversations_file)
    sentence_pairs = create_sentence_pairs(movie_lines, movie_conversations)



    # Tokenize the data
    
    



    tokenized_pairs = []
    for pair in sentence_pairs:
        question_tokens = tokenize_sentence(pair[0])
        answer_tokens = tokenize_sentence(pair[1], is_answer=True)
        tokenized_pairs.append((question_tokens, answer_tokens))
  





    # Filter out the sentences that are too long
   
    # Calculate lengths of all sentences
    all_lengths = [len(sentence.split()) for pair in sentence_pairs for sentence in pair]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=30)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Length of Sentences')
    plt.ylabel('Number of Sentences')
    plt.show()
    
    

    # Choose a suitable max_length based on the histogram
    max_length = 30  
    filtered_pairs = filter_long_sentences(sentence_pairs, max_length)
    
   #saving into pickle
    
   
    # Assuming 'filtered_pairs' is our list of sentence pairs after applying the max_length filter
    filtered_pairs = filter_long_sentences(sentence_pairs, 30)

    # Specify the filename for the pickle file
    pickle_filename = '/Users/chadha/Desktop/HAHI/filtered_sentence_pairs.pkl'

    # Write the filtered pairs to a pickle file
    with open(pickle_filename, 'wb') as pkl_file:
        pickle.dump(filtered_pairs, pkl_file)

    print(f"Filtered sentence pairs have been saved to {pickle_filename}")





   


    # Filter out the words that are too rare
    
    # Step 1: Count the words in your corpus
    word_counts = Counter(word for pair in filtered_pairs for sentence in pair for word in sentence.split())

    # Step 2: Plot the distribution of word frequencies
    frequencies = sorted(word_counts.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies)
    plt.title('Frequency Distribution of Words')
    plt.xlabel('Word Rank')
    plt.ylabel('Frequency')
    plt.show()

    # Step 3: Choose a threshold and eliminate rare words
    threshold = 10000  # This is just an example threshold
    common_words = {word for word, count in word_counts.items() if count >= threshold}

    # Step 4: Filter sentence pairs
    filtered_pairs = [
        pair for pair in filtered_pairs
        if all(word in common_words for sentence in pair for word in sentence.split())
    ]

    # Save the new filtered pairs to a pickle file
    pickle_filename = '/Users/chadha/Desktop/HAHI/filtered_sentence_pairs_no_rare.pkl'
    with open(pickle_filename, 'wb') as pkl_file:
        pickle.dump(filtered_pairs, pkl_file)

    print(f"Filtered sentence pairs without rare words have been saved to {pickle_filename}")
    
    
    
    # Fix the seed for reproducibility
    random_seed = 42
    random.seed(random_seed)

   
    total_pairs = filtered_pairs  

    # Randomly sample 10,000 unique sentence pairs
    sampled_pairs = random.sample(total_pairs, 10000)

    
    # Define paths to data files
    movie_lines_file = Path('/Users/chadha/Desktop/HAHI/movie_lines.txt')
    movie_conversations_file = Path('/Users/chadha/Desktop/HAHI/movie_conversations.txt')

    # Optionally, save the processed data to a pickle file
    pickle_filename = movie_lines_file.parent / 'sampled_sentence_pairs.pkl'
    with open(pickle_filename, 'wb') as pkl_file:
        pickle.dump(sampled_pairs, pkl_file)

    print(f"Sampled sentence pairs have been saved to {pickle_filename}")

   

   
    
    # Assuming sampled_pairs is a list of our sentence pairs
    # Split sampled_pairs into train and validation sets
    split_ratio = 0.8  # for example, 80% for training and 20% for validation
    split_index = int(len(sampled_pairs) * split_ratio)
    
   

    train_pairs = sampled_pairs[:split_index]
    val_pairs = sampled_pairs[split_index:]
    # Quick check to ensure the pairs are structured correctly
    print("Example train pair:", train_pairs[0])
    print("Example val pair:", val_pairs[0])


    

    # Create a Vocabulary instance
    vocab = Vocabulary("English", sampled_pairs)
    # Add words to the vocabulary
    for pair in tokenized_pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])



    # Create Dataset instances for training and validation
    train_dataset = Dataset(vocab, train_pairs)
    val_dataset = Dataset(vocab, val_pairs)

    
    batch_size = 32
    
    
    if batch_size == 1:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda b: train_dataset.collate_fn(b), shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda b: train_dataset.collate_fn(b), shuffle=True)

   
    # Step 2: Train the model
    model = TransformerModel(len(vocab.word2index), 512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, num_heads=8, dropout_p=0.1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    trained_model = train(model, train_dataloader, val_dataloader, epochs=10, criterion=criterion, optimizer=optimizer, scheduler=scheduler, target_loss=1.5)

    # Optionally, train with gradient accumulation
    # trained_model = train_ga(model, train_dataloader, val_dataloader, ...)

    # Step 3: Evaluate the model
    input_sentences = ["How are you today?", "”What is your favorite color?”, ”Do you like music?")

    
    for sentence in input_sentences:
        print(f"Input: {sentence}")
        print("Greedy:", generate_greedy(trained_model, sentence, vocab))
        print("Top-k Sampling:", generate_top_k_sampling(trained_model, sentence, vocab, k=5))
        print("\n")

  
   
  

    
    # Create a Vocabulary instance
    vocab = Vocabulary("English", sampled_pairs)
    
    # Create a Dataset and Dataloader for training and validation
    # You need to split sampled_pairs into training and validation sets
    train_dataset = Dataset(vocab, train_pairs)
    val_dataset = Dataset(vocab, val_pairs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=train_dataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, collate_fn=val_dataset.collate_fn)

    # Step 2: Train the model
    model = TransformerModel(len(vocab.word2index), 512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, num_heads=8, dropout_p=0.1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    trained_model = train(model, train_dataloader, val_dataloader, epochs=10, criterion=criterion, optimizer=optimizer, scheduler=scheduler, target_loss=1.5)

    # Optionally, train with gradient accumulation
    # trained_model = train_ga(model, train_dataloader, val_dataloader, ...)

    # Step 3: Evaluate the model
    input_sentences = ["How are you today?", "Tell me more about artificial intelligence.", "What is the meaning of life?"]
    
    for sentence in input_sentences:
        print(f"Input: {sentence}")
        print("Greedy:", generate_greedy(trained_model, sentence, vocab))
        print("Top-k Sampling:", generate_top_k_sampling(trained_model, sentence, vocab, k=5))
        print("\n")

    
  
    

    pass

