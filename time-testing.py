import time

# Let us check how much time each operation takes 🔽

start_time = time.time()

for i in range(10000):
    print(i*2)

end_time = time.time()

total_time = end_time - start_time
print(f"Total Time taken : {total_time}")
