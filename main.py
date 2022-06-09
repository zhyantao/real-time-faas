from utils import visualization, submit, build_dataset
from models.builder import build_model


def main():
    build_dataset()
    build_model()
    visualization()
    submit()


if __name__ == '__main__':
    main()
