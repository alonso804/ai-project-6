kernel = 3
padding = 1
stride = 2

inputs = 28

for i in range(3):
    inputs = (inputs + padding - kernel) / stride + 1
    print(inputs)
