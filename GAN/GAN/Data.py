import numpy as np

# Dữ liệu (x, y) được tạo thông qua hàm f(x) = 10 + x^2. Như vậy, đồ thị sẽ có dạng parabol. 
def get_y(x):
    return 10 + x*x

# Khởi tạo dữ liệu ngẫu nhiên. 
def sample_data(n=10000, scale=100):
    data = []

    # Tạo mảng số ngẫu nhiên gồm 10000 phần tử trong giới hạn từ 0-0.5 -> 1-0.5
    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)

# Hàm tạo noise ngẫu nhiên. 
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])