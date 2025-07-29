"""
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
pip install .
"""
import numpy as np
import torch 

dim = 512  # 词向量维度
num_elements = 10000
K = 6

np.random.seed(42) 
word_vectors = np.random.randn(num_elements, dim)

word_vectors = torch.rand(num_elements, dim, device="cuda")


import hnswlib
p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction= 200, M=32)
# p.init_index(max_elements=num_elements, ef_construction= 200, M=16, random_seed = 0)
p.add_items(word_vectors.cpu().numpy())

labels, distances = p.knn_query(word_vectors.cpu().numpy(), k=K)
torch.from_numpy(labels.astype(np.int32)).to(word_vectors.device)

# import faiss
# index = faiss.index_factory(dim, "HNSW16", faiss.METRIC_L2)  # HNSW 索引，32 是 M 参数
# index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)  # 转移到 GPU

# index.add(word_vectors.astype('float32')) # 添加库 仅支持浮点数为np.float32格式
# D, I = index.search(word_vectors.astype('float32'), k)  # 搜索最近邻
# print(D, I)
