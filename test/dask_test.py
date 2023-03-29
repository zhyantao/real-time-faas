import dask.dot


@dask.delayed
def inc(x):
    return x + 1


@dask.delayed
def double(x):
    return x * 2


@dask.delayed
def add(x, y):
    return x + y


x = inc(1)
y = inc(2)
z = add(x, y)
w = double(z)

dask.visualize(w)
