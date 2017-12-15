from utils import load_embeddings, pad_and_collate, MonolingualDataset, MonolingualDataLoader

print(1)
embeddings, vocab = load_embeddings('data/en.emb.txt')
print(2)
dataset = MonolingualDataset(folder='data/en', vocab=vocab, train=True)
print(3)
dataloader = iter(MonolingualDataLoader(dataset, batch_size=2, shuffle=True))

for i in range(len(dataloader)):
    print next(dataloader)

