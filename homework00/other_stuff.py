import numpy as np

def meows(n):
    calls = 0
    while calls < n:
        yield "meow " * calls * 2
        calls += 1

if __name__ == "__main__":
    # task 2
    print("\n\r#######################################################")

    liste =  [i*i for i in range(101)]
    print(liste)

    #task 3

    print("\n\r#######################################################") # CATS

    for i in meows(10):
        print(i)

    # task 4
    arr = np.random.normal(0,1,25).reshape((5,5))
    arr[arr > 0.09] = arr * arr
    arr[arr < 0.09] = 42

    print(arr)