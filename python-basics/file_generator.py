from pathlib import Path

def read_large_file(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield line.strip()
            
def make_test_file(n=100_000):
    p=Path("test_data.txt")
    with open(p,"w",encoding="utf-8") as f:
        for i in range(n):
            f.write(f"第{i}行:模拟日志数据\n")
    return p

if __name__=="__main__":
    p=make_test_file()
    count=0
    for line in read_large_file(p):
        count+=1
    print(f"共读取{count}行")