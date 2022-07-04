
# 连续子串 限制子串内部的不同字母的数量
# 滑动窗口
from turtle import right


class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        
        n = len(s)
        
        if n == 0 or k == 0: return 0 

        left, right = 0, 0 
        hashmap = defaultdict()

        max_len = 1 

        while right < n:
            # 改字母的优先级  
            hashmap[s[right]] = right
            right += 1

            if len(hashmap) == k + 1:
                
                # 删除哪个字母
                del_idx = min(hashmap.values())
                del hashmap[s[del_idx]]
                
                # 更新左边界
                left = del_idx + 1 
            # 上述有效  更新答案
            max_len = max(max_len, right-left)
        
        return max_len


# 1023 驼峰式匹配
"""
本质就是选出q中的大写字母  然后与patter比较
"""



"""
998.最大二叉树：
逻辑：
- 如果原二叉树为空  直接根据该节点新建二叉树
- 如果新值大于根节点，则将原来的二叉树插入到新值得左节点
- 如果不大于根节点，则往根节点的右子树上插
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if root == None: return TreeNode(val)

        if root.val < val :
            return TreeNode(val, root, None)

        tmpRight = self.insertIntoMaxTree(root.right, val)
        root.right = tmpRight
        return root

""""
991. 坏了的计算器

只能进行乘2和减1

思路：
- 如果Y大，且是奇数，只能加，偶数，先除
- 如果Y小，只能加

"""
class Solution:
    def brokenCalc(self, startValue: int, target: int) -> int:
        ans = 0
        while target > startValue:
            
            if target % 2 == 0:
                target /= 2 
            else:
                target += 1
            ans += 1

        return ans +  startValue - target



""""
990. 等式方程的可满足性

思路： 
 - 先检查各自的root 如果root不一致  则表示是不同的集合 ， 如果是 == 则将这些合并， 剩下来的 ！= 来否定这些合并的，是否在同一个集合  
 - 如果是在一个集合 则返回False, 如果每个 ！= 都成立 返回True  

"""
class Solution:
    def __init__(self):
        self.father = {}

    def finfRoot(self, node):
        tmp = node
        
        while tmp != self.father[tmp]:
            
            tmp = self.father[tmp]
        
        self.father[node] = tmp

        return tmp


    def equationsPossible(self, equations: List[str]) -> bool:
        equal, noEqual = [], []
        
        for s in equations:
            a, relation, b = s[0], s[1:-1], s[-1]
            print(a, relation, b)
            self.father[a] = a 
            self.father[b] = b 
            
            if relation == '==' :
                equal.append([a, relation, b])
            else:
                noEqual.append([a, relation, b])
    
        for a, relation, b in equal:
            rootA = self.finfRoot(a)
            rootB = self.finfRoot(b)
            self.father[rootA] = rootB
        
        for a, relation, b in noEqual:
            rootA = self.finfRoot(a)
            rootB = self.finfRoot(b)

            if rootA == rootB:
                return False 
        return True



"""
988. 从叶结点开始的最小字符串

思路：
将所有的答案计算出来，比较字典序
将答案转换成字母

注释：
 - 常见的函数： 
    ord 将字母转换成整数
    chr 将整数转换成字母
"""

class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        
        self.ans = "z" * 100

        def dfs(node, path):
            if node == None: return 
            
            if node.left == None and node.right == None: 
                path += (chr(ord('a')  + node.val))
                
                if path[::-1] < self.ans: 
                    self.ans = path[::-1]
                return

            dfs(node.left, path + (chr(ord('a')  + node.val)))
            dfs(node.right, path + (chr(ord('a')  + node.val)))

            return 

        # print(chr(ord('a')  + root.val) )

        dfs(root, "")

        return self.ans


"""

985. 查询后的偶数和
思路：跟着题目意思来

"""

class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        preSum = sum([i for i in nums if i % 2 == 0])
        ans = []
        for delta, index in queries:
            
            if nums[index] % 2 == 0:
                preSum -= nums[index]
            
            nums[index] += delta

            if nums[index] % 2 == 0:
                preSum += nums[index]
            ans.append(preSum)
        return ans



"""
984. 不含 AAA 或 BBB 的字符串
思路：跟着题目意思来
    难点： 贪心的构建合适的字符串
"""

class Solution:
    def strWithout3a3b(self, A: int, B: int) -> str:

        ans = []

        while A or B:
            if len(ans) >= 2 and ans[-1] == ans[-2]:
                writeA = ans[-1] == 'b'
            else:
                writeA = A >= B 

            if writeA :
                A -= 1
                ans.append('a')
            else:
                B -= 1
                ans.append(b)

            if A == B:
                if ans[-1] != 'a':
                    ans.append('ab' * A)
                    # A = B = 0
                else:
                    ans.append('ba' * A)
                A = B = 0 


        return "".join(ans)           
        
"""
983. 最低票价
方法一：记忆化搜索（日期变量型）;
思路和算法;
dp(i)=min{cost(j)+dp(i+j)},j∈{1,7,30}
最终的答案记为 dp(1)。
"""
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        dayset = set(days)
        duration = [1, 7 , 30]

        @lru_cache(None)
        def dp(i):
            if i > 365:
                return 0 
            elif i in days:
                return min( dp(i+d) + c ) for c, d in zip(costs, duration)
            else:
                return dp(i+1)
        return dp(1) 




""""
355. 设计推特 

这题做的有问题

"""

class Twitter:

    def __init__(self):
        self.findFans = defaultdict(list)
        self.findStars = defaultdict(list)
        self.id2news = defaultdict(deque)
                
                
    def postTweet(self, userId: int, tweetId: int) -> None:
        # userPreTweetId = 
        # 修改自身的news
        while len(self.id2news[userId]) >= 10:
            self.id2news[userId].popleft()
        # else:
        self.id2news[userId].append(tweetId)
        
        # 修改funs的news
        # 粉丝ID list
        funsIDs = self.findFans[userId]
        
        # 挨个修改粉丝的推文 
        for funs in funsIDs:
            # 找到粉丝的之前推文
            # preNews = self.id2news[funs]
            while self.id2news[funs] >= 10:
                self.id2news[funs].popleft()
            
            self.id2news[funs].append(tweetId)


    def getNewsFeed(self, userId: int) -> List[int]:
        return self.id2news[userId]        


    def follow(self, funs: int, star: int) -> None:
        self.findFans[star].append(funs)
        self.findStars[funs].append(star)

    def unfollow(self, funs: int, star: int) -> None:
        # ans = []
        # for i in self.findStars[funs]:
        #     if i != 
        ans = [ i for i in self.findStars[funs] if i != star  ]
        self.findStars[funs] = ans

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)



"""
128 最长连续序列
"""

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = list(set(nums))
        nums.sort()
        left, right = 0 , 0 
        n = len(nums)
        ans = 0
        print(nums)

        while left < n and right < n:
            right = left + 1 
            while right < n and nums[right] - nums[right-1]  == 1:
                right  += 1 
            ans = max(ans, right - left)
            # print(left, right)
            left = right
        return ans



"""
117  填充每个节点的下一个右侧节点指针 II
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        que = deque()
        
        if root == None: return None
        que.append(root)

        while que:
            n = len(que)
            # tmpNodeList = []
            preTmpNode = None

            for i in range(n):
                
                tmpNode = que.popleft()
                # tmpNodeList.append(tmpNode)
                tmpNode.next = preTmpNode
                preTmpNode = tmpNode

                if tmpNode.right:
                    que.append(tmpNode.right)
                
                if tmpNode.left:
                    que.append(tmpNode.left)

        return root


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

"""
109. 有序链表转换二叉搜索树

给定一个单链表的头节点  head ，其中的元素 按升序排序 ，将其转换为高度平衡的二叉搜索树。
本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差不超过 1。

思路： 像这种题 最好拆分成递归的写法
核心： 找到中心点 以其作为根节点  
       前面的构建平衡二叉树
       后面的构建平衡二叉树

"""
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:

        if head == None or head.next == None: return head
        if head.next.next == None: 
            left = TreeNode(head.val, None, None)
            right = None
            root = TreeNode(head.val, left, right)
            return root
        
        pre, left, right = head, head, head

        while right != None and right.next != None and right.next.next != None:
            pre = left
            left = left.next
            right = right.next.next

        pre.next = None
        # root = TreeNode
        leftTree = self.sortedListToBST(head)
        rightTree = self.sortedListToBST(left.next)
        rootNode = TreeNode(left.val, leftTree, rightTree)

        return rootNode

"""
1503. 所有蚂蚁掉下来前的最后一刻

因为碰撞的那一时刻，不花费任何时间，所以等价于那只蚂蚁接着往前走
位置p的蚂蚁，left方向需要的时间是：p 
            rigtht方向需要的时间是: n-p 
"""

class Solution:
    def getLastMoment(self, n: int, left: List[int], right: List[int]) -> int:
        ans = float("-inf")
        for i in left:
            ans = max(ans, i)
        
        for i in right:
            ans = max(ans, n - i)
        return ans



"""
1619. 删除某些元素后的数组均值
"""

class Solution:
    def trimMean(self, nums: List[int]) -> float:
        nums.sort()
        n = len(nums)
        left = round(n * 0.05)
        right = n -left
        ans = sum(nums[left:right]) / (right - left)
        return ans


"""
973. 最接近原点的 K 个点
"""

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        ans = []
        for a, b in points:
            ans.append([a,b,((a * a) + (b * b))])
        
        ans.sort(key=lambda x:x[2])
        return [[a, b] for a,b,_ in ans[:k]]


"""
1026. 节点与其祖先之间的最大差值
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # def inorder(self, )
    def __init__(self) -> None:
        self.minVle, self.maxVle = float('-inf'), float("inf")
        
        
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        # 换一种思路的话：改点与前面的点的最大落差：前面的最大值和最小值
        if root != None:
            if root.val < self.minVle:
                self.minVle = root.val

            if root.val > self.maxVle:
                self.maxVle = root.val

        if root == None or (root.left == None or root.right == None):
            return 0

        left = self.maxAncestorDiff(root.left)
        right = self.maxAncestorDiff(root.right)

        ans = max(abs(root.val - self.maxVle), abs(root.val - self.minVle), left, right)
        return ans
