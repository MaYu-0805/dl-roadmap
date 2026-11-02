import asyncio
import time

async def fetch(name,delay):
    print(f"{name} 开始")
    await asyncio.sleep(delay)
    print(f"{name} 完成")
    return f"{name} 的结果"

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(
        fetch("任务A", 1),
        fetch("任务B", 2),
        fetch("任务C", 3),
    )
    print(results)
    print(f"总耗时: {time.perf_counter() - start:.2f} s")

if __name__ == "__main__":
    asyncio.run(main())