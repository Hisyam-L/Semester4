import time
import random
from datetime import datetime

def bubble_sort(arr):
    start_time = time.time()
    start_clock = datetime.now().strftime("%H:%M:%S")
    n = len(arr)
    # Optimasi: Jika dalam satu putaran tidak ada pertukaran, artinya sudah urut
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    end_clock = datetime.now().strftime("%H:%M:%S")
    akhir_masa = time.time()
    durasi = akhir_masa - start_time
    print(f"Start time : {start_clock} , akhir time : {end_clock} dengan waktu : {durasi}")

if __name__ == "__main__":
    data0 = [random.randint(1, 1000000) for _ in range(10)]
    data1= [random.randint(1, 1000000) for _ in range(100)]
    data2 = [random.randint(1, 1000000) for _ in range(1000)]
    data3 = [random.randint(1, 1000000) for _ in range(10000)]
    data4 = [random.randint(1, 1000000) for _ in range(20000)]
    data5 = [random.randint(1, 1000000) for _ in range(30000)]
    data6 = [random.randint(1, 1000000) for _ in range(40000)]
    data7 = [random.randint(1, 1000000) for _ in range(50000)]
    data8 = [random.randint(1, 1000000) for _ in range(60000)]
    data9 = [random.randint(1, 1000000) for _ in range(70000)]
    data10 = [random.randint(1, 1000000) for _ in range(80000)]
    data11 = [random.randint(1, 1000000) for _ in range(90000)]
    data12 = [random.randint(1, 1000000) for _ in range(100000)]

    bubble_sort(data0)
    bubble_sort(data1)
    bubble_sort(data2)
    bubble_sort(data3)
    bubble_sort(data4)
    bubble_sort(data5)
    bubble_sort(data6)
    bubble_sort(data7)
    bubble_sort(data8)
    bubble_sort(data9)
    bubble_sort(data10)
    bubble_sort(data11)
    bubble_sort(data12)



