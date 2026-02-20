from datasets import load_dataset

# Load PMC Open Access dataset
dataset = load_dataset("pmc/open_access", split="train")

print("Dataset loaded successfully!")
print(dataset[0])   # Show first article
