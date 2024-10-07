import torch

# 创建一个 3x3 的张量
tensor = torch.rand(3, 3)
print("Original Tensor:")
print(tensor)

# 张量加法
tensor_add = tensor + tensor
print("\nTensor after addition:")
print(tensor_add)

# 张量乘法
tensor_mul = tensor * tensor
print("\nTensor after multiplication:")
print(tensor_mul)