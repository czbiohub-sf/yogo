import mutliprocessing as np

from pathlib import Path


def normalize(x, y):
    height = 600
    width = 800
    return float(x) / width, float(y) / height


def convert_coords(xmin, xmax, ymin, ymax):
    xmin, ymin = normalize(xmin, ymin)
    xmax, ymax = normalize(xmax, ymax)

    if xmin == xmax or ymin == ymax:
        raise ValueError(
            f"xmin == xmax = {xmin == xmax} or ymin == ymax = {ymin == ymax}"
        )

    assert xmin < xmax, f"need xmin < xmax, got {xmin}, {xmax}"
    assert ymin < ymax, f"need ymin < ymax, got {ymin}, {ymax}"

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    assert all(
        [0 < v < 1 for v in [xcenter, ycenter, width, height]]
    ), f"{[xcenter, ycenter, width, height]}"

    return xcenter, ycenter, width, height


def multiprocess_directory_work(files, work_fcn):
    cpu_count = mp.cpu_count()

    print(f"processing {len(files)} files")
    print(f"num cpus: {cpu_count}")

    with mp.Pool(cpu_count) as P:
        # iterate so we get tqdm output, thats it!
        for _ in tqdm(
            P.imap_unordered(work_fcn, files, chunksize=64), total=len(files)
        ):
            pass
        P.close()
        P.join()
