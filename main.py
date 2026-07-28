from internal.entrypoints.gspl import cli
import time

if __name__ == "__main__":
    start = time.time()
    cli()
    end = time.time()
    print(f"Time: {end-start} sec")
