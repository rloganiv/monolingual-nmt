from utils import load_embeddings, pad_and_collate, MonolingualDataset, MonolingualRandomSampler
from torch.utils.data import DataLoader

embeddings, vocab = load_embeddings('test_embeddings.txt')
dataset = MonolingualDataset(folder='test', vocab=vocab, train=True)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=pad_and_collate,
                        sampler=MonolingualRandomSampler(dataset))

for i, batch in enumerate(dataloader):
    print i, batch

