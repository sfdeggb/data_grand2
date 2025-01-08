import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1

    pos = np.array(np.arange(position), np.float32)
    ind = np.array(np.arange(d_model), np.float32)
    pos = pos.reshape(position, 1)
    ind = ind.reshape(1, d_model)

    def get_angles(pos, i):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angles

    angle1 = get_angles(pos, ind)
    sine = np.sin(angle1[:, 0::2])
    cosine = np.cos(angle1[:, 1::2])
    pos_encoding = np.concatenate([sine, cosine], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, :]
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding
if __name__ == "__main__":
    print(pos_encoding(2, 8))
    