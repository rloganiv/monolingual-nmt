import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from beam_search import Beam
from nmt_mono import load_config, transform_inputs
from model_remake import Model
from torch.autograd import Variable
from utils import greedy_translate, load_embeddings, MonolingualDataset, MonolingualDataLoader


USE_CUDA = torch.cuda.is_available()

# Converts list of idxs to string of words
def idxs_to_string(idxs, idx2word):
	# Covert from idx to word if not beginning of sentence or padding
	bos = '<s>'
	eos = '</s>'
	words = [idx2word[idx] for idx in idxs if (idx2word[idx] != bos) and (idx2word[idx] != eos)]
	return ' '.join(words)

"""
	Args:
		model - encoder/decoder model
		src - sentence from source langauge (batch size 1)
		src_lang - 'l1' or 'l2'
		lengths - length of src
		max_trg_len - maximum length of decoded sentence
		verbose - at each decoding step, prints out highest probability sentence
	Returns:
		Translated sentence (string)
"""
def translate(model, src, src_lang, lengths, beam_size, max_trg_len, target_vocab, verbose=False):
    # Use decoder and generator opposite the source langauge
    if src_lang == 'l1':
        decoder = model.l2_decoder
        generator = model.l2_to_vocab
    elif src_lang == 'l2':
        decoder = model.l1_decoder
        generator = model.l1_to_vocab
    else:
        raise ValueError('src_lang')
        
    enc_hidden, context = model.encoder(src, src_lang, lengths)
    dec_state = decoder.init_decoder_state(src, context, enc_hidden)

    beam = Beam(beam_size, target_vocab._word2idx, dec_state)
    idx2word = target_vocab._idx2word
    vocab_size = len(idx2word)
    
    logsoftmax = torch.nn.LogSoftmax() # Use log softmax since beam search code adds probabilities
    
    done = False
    depth = 0
    while (not done) and (depth < max_trg_len):
        depth += 1
        next_word_lk = torch.zeros((beam_size, vocab_size)) # Likelihood of next words
        next_dec_states = []
        cur_words = beam.get_current_state() # words in beam 
        cur_dec_states = beam.get_current_decoder_state() # to feed into decoder
        
         # Decode one step for each word in beam
        for i in range(beam_size):
            word = cur_words[i]
            dec_state = copy.copy(cur_dec_states[i])
            if word == beam.pad:
                next_dec_states.append(None)
            else:
                word = Variable(torch.LongTensor([word])).unsqueeze(0).unsqueeze(2)
                dec_out, dec_state, _ = decoder(word, context, dec_state,
                                               context_lengths=lengths) 
                preds = generator(dec_out)
                preds_prob = logsoftmax(preds.squeeze(1)) 
                # Probabilities of next words 
                next_word_lk[i,:] = preds_prob.data
                # Save decoder states
                next_dec_states.append(dec_state)
        done = beam.advance(next_word_lk, next_dec_states)
        if verbose: # Print top sentence at each time point
            print idxs_to_string(beam.get_hyp(0), idx2word)
    return idxs_to_string(beam.get_hyp(0), idx2word)

# # Example translation
# if __name__ == '__main__':
#     # Load config
# 	config_file = 'config_en_fr.json'
# 	config = load_config(config_file)

# 	# Load saved model 
# 	print "Loading model"
# 	model = torch.load('ckpt-bt/model.pt').cpu()
# 	model.dropout = 0
# 	model.encoder.rnn.dropout = 0
# 	model.l1_decoder.dropout = torch.nn.Dropout(0)
# 	model.l1_decoder.rnn.dropout = torch.nn.Dropout(0)
# 	model.l2_decoder.dropout = torch.nn.Dropout(0)
# 	model.l2_decoder.rnn.dropout = torch.nn.Dropout(0)

# 	# Load embeddings and (test) datasets
# 	l1_embeddings, l1_vocab = load_embeddings(path=config.data.l1_embeddings)
# 	l2_embeddings, l2_vocab = load_embeddings(path=config.data.l2_embeddings)
# 	l1_dataset = MonolingualDataset(folder=config.data.l1_test_data,
# 	                                train=False, 
# 	                                vocab=l1_vocab)
# 	l2_dataset = MonolingualDataset(folder=config.data.l2_test_data,
# 	                                train=False,
# 	                                vocab=l2_vocab)

# 	# Translate 10 sentences from test set from l2->l1
# 	# Also time results
# 	start = time.time()
# 	beam_size = 12
# 	for i in range(10): 
# 		# Hacks needed to get test batches of size 1
# 		l2_sample = l2_dataset[i]
# 		l2_src = Variable(torch.LongTensor(l2_sample['src'])).unsqueeze(0)
# 		l2_lengths = torch.LongTensor([l2_sample['src_len']])
# 		# End hack
# 		l2_src, l2_lengths, _, l2_index = transform_inputs(
# 			src=l2_src,
# 			lengths=l2_lengths,
# 			tgt=l2_src)

# 		print translate(model=model, 
# 	    	src=l2_src, 
# 	    	src_lang='l2', 
# 	    	lengths=l2_lengths, 
# 	    	beam_size=beam_size,
# 	    	max_trg_len=config.data.max_length, 
# 	    	target_vocab=l1_vocab) # MAKE SURE target_vocab IS TARGET, NOT SOURCE VOCAB
# 	print time.time() - start



# Translate test set
if __name__ == '__main__':
    # Load config
	config_file = 'config_en_fr.json'
	config = load_config(config_file)

	# Load saved model 
	print "Loading model"
	model = torch.load('ckpt-bt/model.pt').cpu()
	model.eval()

	# Load embeddings and (test) datasets
	l1_embeddings, l1_vocab = load_embeddings(path=config.data.l1_embeddings)
	l2_embeddings, l2_vocab = load_embeddings(path=config.data.l2_embeddings)

	# Translate all test files
	start = time.time()
	beam_size = 12

	test_dirs = ['ref_en', 'ref_fr', 'src_en', 'src_fr']
	test_dirs = ['test_data/' + f for f in test_dirs]

	for test_dir in test_dirs:
	    src_lang = test_dir.split("_")[2]
	    if src_lang == 'en':
	        src_lang = 'l1'
	        src_vocab = l1_vocab
	        trg_lang = 'l2'
	        trg_vocab = l2_vocab
	    elif src_lang == 'fr':
	        src_lang = 'l2'
	        src_vocab = l2_vocab
	        trg_lang = 'l1'
	        trg_vocab = l1_vocab
	    else:
	        ValueError('source language')
	    
	    test_dataset = MonolingualDataset(folder=test_dir, vocab=src_vocab)
	    test_file = test_dataset._paths[0].split('/')[2]
	    print test_file, src_lang
	    f = open('test_translated/' + test_file + '_translated', 'w')
	    for i in range(test_dataset._line_counts[0]):
	        sample = test_dataset[i]
	        # Hacks needed to get test batches of size 1
	        src = Variable(torch.LongTensor(sample['src'])).unsqueeze(0)
	        lengths = torch.LongTensor([sample['src_len']])
	        # End hack
	        src, lengths, _, _ = transform_inputs(
	            src=src,
	            lengths=lengths,
	            tgt=src)
	        translated = translate(model=model, 
	            src=src, 
	            src_lang=src_lang, 
	            lengths=lengths, 
	            beam_size=beam_size,
	            max_trg_len=config.data.max_length, 
	            target_vocab=trg_vocab)
	        f.write(translated+'\n')
	    f.close()
	    print("Time to translate file (secs): ",  time.time() - start)



