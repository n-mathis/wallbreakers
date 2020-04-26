from collections import defaultdict, Counter, deque
from heapq import heappush, heappop, heapreplace, heapify
import math

class Node:

    def __init__(self, k, v):
        """
        :type k: int
        :type v: int
        """
        self.key = k
        self.val = v
        self.prev = None
        self.next = None

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.dic = dict()
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.dic: # if the key exists
            n = self.dic[key] # get the node
            self.removeNode(n) # remove it from the linked list
            self.addNode(n) # readd it since it's now the most recent
            return n.val
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.dic: # if the key exists
            self.removeNode(self.dic[key]) # remove it from the linkedlist
        n = Node(key, value)
        self.addNode(n) # add new Node to linked list
        self.dic[key] = n # add it to the dictionary
        if len(self.dic) > self.capacity: # if cache/linked list is at capacity
            n = self.head.next # remove the head (the oldest node)
            self.removeNode(n)
            del self.dic[n.key] # delete it from the dictionary
            
    def removeNode(self, node):
        """
        :type node: Node
        """
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p
        
    def addNode(self, node):
        """
        :type node: Node
        """
        p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail

class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.deque = deque()
        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        self.deque.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.deque.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.deque[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return not self.deque

class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.deque = deque()

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.deque.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.deque.popleft()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.deque[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not self.deque

class Solution(object):

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        while head != None:
            curr = head
            head = head.next
            curr.next  = prev
            prev = curr
        return prev
    
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        curr = head
        prev = dummy
        for x in range(m - 1):
            curr = curr.next
            prev = prev.next
        for x in range(n-m):
            temp = curr.next
            curr.next = temp.next
            temp.next = prev.next
            prev.next = temp

        return dummy.next

    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return head
        odd = head
        even = head.next
        evenHead = even
        while even != None and even.next != None:
            odd.next = odd.next.next
            odd = odd.next
            even.next = even.next.next  
            even = even.next
        odd.next = evenHead
        return head
        
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA and headB:
            A = headA
            B = headB
            while A != B:
                if A:
                    A = A.next 
                elif not A:
                    A = headB
                if B:
                    B = B.next
                elif not B:
                    B = headA
            return A
    
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        jump = ListNode(-1)
        dummy = jump
        dummy.next = head
        l = head
        r = head
        while True:
            count = 0
            while r and count < k:
                count += 1
                r = r.next
            if count == k:
                prev = r
                curr = l
                for x in range(k):
                    temp = curr.next
                    curr.next = prev
                    prev = curr
                    curr = temp
                jump.next = prev
                jump = l
                l = r
            else:
                return dummy.next

    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        stack = []
        for i in range(len(ops)):
            if ops[i] == "C":
                stack.pop()
            elif ops[i] == "D":
                stack.append(stack[-1]*2)
            elif ops[i] == "+":
                stack.append(stack[-1]+stack[-2])
            else:
                stack.append(int(ops[i]))
        return sum(stack)

    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        dic = {}
        stack = []
        for x in nums2:
            while stack and stack[-1] < x:
                dic[stack.pop()] = x
            stack.append(x)
        result = [-1]*len(nums1)
        for i,x in enumerate(nums1):
            if x in dic:
                result[i] = dic[x]
        return result

    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        back = lambda res, c: res[:-1] if c == '#' else res + c
        return reduce(back, S, "") == reduce(back, T, "")

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        dic = {"]":"[", "}":"{", ")":"("}
        for c in s:
            if c in dic.values():
                stack.append(c)
            elif c in dic.keys():
                if stack == [] or dic[c] != stack.pop():
                    return False
            else:
                return False
        return stack == []

    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        stack = [0]
        for c in S:
            if c == "(":
                stack.append(0)
            else:
                last = stack.pop()
                stack[-1] += max(2*last, 1)
        return stack.pop()

    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        j = 0
        while n > 0 and k % n != 0:
            for i in range(0, k):
                nums[j+i], nums[len(nums)-k+i] = nums[len(nums)-k+i], nums[j+i]
            n = n-k
            j = j+k
            k = k % n

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        freq = []
        counter = Counter(nums)
        # since it will be a min heap but we want the most frequent we switch them to negative
        h = [(-count, num) for num, count in counter.items()] 
        heapify(h)
        while len(freq) < k:
            freq.append(heappop(h)[1])
        return freq

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        node = ListNode(None)
        dummy = node
        h = [(n.val, n) for n in lists if n] # list of head nodes
        heapify(h)
        while h:
            v, n = h[0] # get the min in the heap, the value and the head
            if n.next is None:
                heappop(h) # None means end of list we're on, ignore it
            else:
                heapreplace(h, (n.next.val, n.next)) # pop min node and push the min node's next
            node.next = n # linking each min node so it's sorted
            node = node.next

        return dummy.next
    
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        for num in nums:
            result += [i + [num] for i in result]
        return result
    
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        return reduce(lambda P, n: [p[:i]+[n]+p[i:] for p in P for i in range(len(p)+1)], nums, [[]])
    
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        combs = [[]]
        for x in range(k):
            combs = [[i]+c for c in combs for i in range(1, c[0] if c else n+1)]
        return combs
    
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        dp = [[] for i in range(n+1)]
        dp[0].append('')
        for i in range(n+1):
            for j in range(i):
                dp[i] += ['('+x+')'+y for x in dp[j] for y in dp[i-j-1]]
        return dp[n]

    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = [0] 
        for i in range(n):
            for j in range(len(res)-1, -1, -1):
                res.append(res[j] | 1<<i)
        return res 

    def allPossibleFBT(self, N):
        """
        :type N: int
        :rtype: List[TreeNode]
        """
        N -= 1
        if N == 0: return [TreeNode(0)]
        ret = []
        for l in range(1, N, 2):
            for left in self.allPossibleFBT(l):
                for right in self.allPossibleFBT(N - l):
                    root = TreeNode(0)
                    root.left = left
                    root.right = right
                    ret += [root]
        return ret
    
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        self.dfsCombSum(candidates, target, 0, [], res)
        return res
    
    def dfsCombSum(self, candidates, target, index, path, res):
        if target < 0:
            return res
        if target == 0:
            res.append(path)
            return res
        for i in range(index, len(candidates)):
            self.dfsCombSum(candidates, target-candidates[i], i, path+[candidates[i]], res)
    
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        total = 0
        for num in nums:
            total += num
        if (total%2) == 1:
            return False
        half = total/2
        n = len(nums)
        dp = [False] * (half+1)
        dp[0] = True
        for num in nums:
            i = half
            while i > 0:
                if i >= num:
                    dp[i] = dp[i] or dp[i-num]
                i -= 1
        return dp[half]
    
    def canPartitionKSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        nums.sort(reverse=True)
        parts = [0] * k
        kSum = sum(nums) // k
        return self.dfsKSubsets(0, nums, k, parts, kSum)
    
    def dfsKSubsets(self, ind, nums, k, parts, kSum):
        if ind == len(nums):
            return True
        for i in range(k):
            parts[i] += nums[ind]
            if parts[i] <= kSum and self.dfsKSubsets(ind+1, nums, k, parts, kSum):
                return True
            parts[i] -= nums[ind]
            if parts[i] == 0:
                break
        return False
    
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        self.solveFromCell(0, 0, board)
    
    def solveFromCell(self, row, col, board):
        if col == len(board[row]):
            col = 0
            row += 1
            if row == len(board):
                return True
        if board[row][col] != '.':
            return self.solveFromCell(row, col+1, board)
        for value in range(1,len(board)+1):
            val = str(value)
            if self.canPlaceVal(board, row, col, val):
                board[row][col] = val
                if self.solveFromCell(row, col+1, board):
                    return True
                board[row][col] = '.'
        return False
    
    def canPlaceVal(self, board, row, col, val):
        for r in board:
            if val == r[col]:
                return False
        for i in range(len(board[row])):
            if val == board[row][i]:
                return False
        boxSize = int(math.sqrt(len(board)))
        topLeftBoxRow = boxSize * (row/boxSize)
        topLeftBoxCol = boxSize * (col/boxSize)
        for i in range(boxSize):
            for j in range(boxSize):
                if val == board[topLeftBoxRow+i][topLeftBoxCol+j]:
                    return False
        return True