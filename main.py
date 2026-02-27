from internal.entrypoints.gspl import cli
import time

if __name__ == "__main__":
    start = time.time()
    cli()
    end = time.time()
    print(f"运行时间: {end-start}秒")
