def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    print(*a)
    for i in zip(*a):
        print(i)
    return [list(i) for i in zip(*a)]
if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    print(transpose_matrix(a))