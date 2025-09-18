from loader import train_dl, test_dl

inputs, target = next(iter(train_dl))
ratio_water = target.sum().float() / target.numel()
print(f"fire fraction = {ratio_water:.4f}")

# class weights: w = (1 - p) / p  where p = flood fraction
p   = ratio_water.item()
w0  = 1.0                       # weight for “no flood”
w1  = (1 - p) / p               # weight for “water”
print(w1)

print(target.shape)
print(target.unique())      # should print tensor([0, 1]) for binary
print(target.dtype)         # should be torch.int64 (long)
print(target.min(), target.max())