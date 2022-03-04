from collections import defaultdict

class UnionFind(object):
    def __init__(self, names):
        self.parent = {}
        for name in names:
            self.parent[name] = name

    def union(self, a, b):
        if a not in self.parent:
            self.parent[a] = a
        if b not in self.parent:
            self.parent[b] = b
        root_a = self.find_root(a)
        root_b = self.find_root(b)
        if root_a < root_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_a] = root_b

    def find_root(self, node):
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

class Solution(object):
    def trulyMostPopular(self, names, synonyms):
        """
        leetcode17.07
        :type names: List[str]
        :type synonyms: List[str]
        :rtype: List[str]
        """
        # 频率map
        freq_map = defaultdict(int)
        for name_freq in names:
            name, freq_str = (part.strip().strip(')') for part in name_freq.split('('))
            freq_map[name] = int(freq_str)
        # 初始化并查集
        uf = UnionFind(freq_map.keys())
        # 并操作
        for pair_str in synonyms:
            a, b = (name.strip().strip(')').strip('(') for name in pair_str.split(','))
            uf.union(a, b)
        # 生成结果
        result = []
        res_map = defaultdict(int)
        for name, freq in freq_map.items():
            res_map[uf.find_root(name)] += freq
        for name, freq in res_map.items():
            result.append('{}({})'.format(name, freq))
        return result
