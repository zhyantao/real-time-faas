from models.ModelParallelTest import model_parallel_test
from models.MultiCudaTest import multi_cuda_test
from models.SingleCudaTest import single_cuda_test


def build_model():
    # single_cuda_test()
    # multi_cuda_test()
    model_parallel_test()
