import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        cost = time.perf_counter()-start
        print(f"[timer] {func.__name__} 耗时 {cost:.4f}s")
        return result
    return wrapper

@timer
def train(n):
    s=0
    for i in range(n):
        s+=i
    return s

if __name__=="__main__":
    print(train(10_000_000))
    print(train.__name__)