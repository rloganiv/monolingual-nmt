from utils import load_embeddings, pad_and_collate, MonolingualDataset, MonolingualDataLoader

embeddings, vocab = load_embeddings('test_embeddings.txt')
dataset = MonolingualDataset(folder='test', vocab=vocab, train=True)
dataloader = iter(MonolingualDataLoader(dataset, batch_size=2, shuffle=True))

for i in range(len(dataloader)):
    print next(dataloader)

