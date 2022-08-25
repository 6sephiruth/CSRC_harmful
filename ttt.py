import pickle
import time


ddd = [220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704]

for date in ddd:

    x = pickle.load(open(f'./result/{date}_name','rb'))
    y = pickle.load(open(f'./result/{date}_value','rb'))

    print(x[:500])
    print(y[:500])

    print()
    print()
    print()

    time.sleep(10)
