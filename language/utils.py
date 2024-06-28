import pickle

''' 创建读取函数 '''
def create_read_func(vocab_path):
    # 用于从给定的路径 vocab_path 读取并反序列化数据
    def read_func():
        with open(vocab_path, 'rb') as f:
            # 使用了 Python 的 pickle 模块来加载存储的对象
            data = pickle.load(f)
        return data

    return read_func


def create_write_func(vocab_path):
    def write_func(data):
        with open(vocab_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return write_func
