import pickle

with open("models/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Keys:", list(data.keys())[:5])

for name, embedding in list(data.items())[:3]:
    print(name, type(embedding), getattr(embedding, "shape", None))