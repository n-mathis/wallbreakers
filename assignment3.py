from collections import defaultdict

class Solution(object):

    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        rightmost = {c:i for i, c in enumerate(S)}
        left, right = 0, 0
        result = []
        for i, letter in enumerate(S):
            right = max(right,rightmost[letter])
            if i == right:
                result += [right-left + 1]
                left = i+1
        return result

    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        points = sorted(points, key = lambda x: x[1])
        res = 0
        end = -float('inf')
        for interval in points:
            if interval[0] > end:
                res += 1
                end = interval[1]
        return res

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        for c in s:
            i = t.find(c)
            if i == -1:
                return False
            t = t[i+1:]
        return True

    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort()
        s.sort()
        i = 0
        j = 0
        while i<len(g) and j<len(s):
            if s[j] >= g[i]:
                i += 1
            j +=1 
        return i

    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        five = 0
        ten = 0
        for i in bills:
            if i == 5:
                five += 1
            elif i == 10:
                five -= 1
                ten += 1
            elif ten > 0:
                five -= 1
                ten -= 1
            elif i == 20:
                five -= 3
            if five < 0: 
                return False
        return True

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        return -1

    def peakIndexInMountainArray(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        left = 0
        right = len(A) - 1
        peak = 0
        while left <= right:
            mid = (left+right) // 2
            if A[mid-1] < A[mid] > A[mid+1]:
                peak = mid
                return peak
            elif A[mid-1] < A[mid]:
                left = mid + 1
            else:
                right = mid - 1

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s) == sorted(t)

    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        i = 0
        total = 0
        while i < len(nums):
            total += nums[i]
            i += 2
        return total

    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if intervals == []:
            return intervals
        intervals = sorted(intervals, key = lambda x: x[0])
        res = [intervals[0]]
        for interval in intervals[1:]:
            if interval[0] <= res[-1][1]:
                res[-1][1] = max(interval[1], res[-1][1])
            else:
                res.append(interval)
        return res

    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if len(p) == 0 or len(s) == 0:
            return []
        sP = sorted(p)
        sS = sorted(s[:len(p)])
        startIs = set()
        for ind in range(len(s)-len(p)+1):
            if sP == sS:
                startIs.add(ind)
            sub = s[ind]
            del sS[bisect.bisect_left(sS, sub)]
            try:         
                add = s[ind +len(p)]
                bisect.insort_left(sS, add)
            except:
                break
        if sP == sS:
            startIs.add(ind)
        return list(startIs)

    def myPowR(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        elif n < 0:
            return 1 / self.myPowR(x, -n)
        return x * self.myPowR(x, n-1)
    
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        elif n < 0:
            return 1 / self.myPow(x, -n)
        elif n%2 == 0:
            return self.myPow(x*x, n//2)
        return x * self.myPow(x, n-1)

    def maxProfitHelper(self, x, prices):
        if x == 0:
            return 0
        else:
            profit = self.maxProfitHelper(x - 1, prices)
            for i in range(0, x):
                profit = max(profit, prices[x-1] - prices[i])
            return profit

    def maxProfitR(self, prices):
        return self.maxProfitHelper(len(prices), prices)

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if prices == []:
            return 0
        dp = [0] * len(prices)
        minP = prices[0]
        for i in range(len(prices)):
            dp[i] = max(dp[i-1], prices[i]-minP)
            minP = min(minP, prices[i])
        return dp[-1]
    
    def minDistanceR(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        memo = {}
        if word1 == word2:
            return 0
        s, t = sorted([word1, word2], key=len)
        if not s:
            return len(t)
        if (s, t) in memo:
            return memo[(s, t)]
        i = 0
        while i < len(s) and s[i] == t[i]:
            i += 1
        if i > 0:
            memo[(s, t)] = self.minDistanceR(s[i:], t[i:])
        else:
            memo[(s, t)] = 1 + min(self.minDistanceR(s[1:], t), 
                                   self.minDistanceR(s, t[1:]), 
                                   self.minDistanceR(s[1:], t[1:]))
        return memo[(s, t)]
    
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        dp = [[0]*(len(word2)+ 1) for x in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+ 1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[-1][-1]

    def robHelperR(self, memo, nums, f, l):
        """
        :type memo: {}
        :type nums: List[int]
        :type f: int
        :type l: int
        :rtype: int
        """
        if f == l:
            return nums[f]
        if f > l:
            return 0
        if memo[(f,l)] == -1:
            memo[(f,l)] = max(self.robHelperR(memo, nums, f+2, l) + nums[f], self.robHelperR(memo, nums, f+1, l))
        return memo[(f,l)]
        
    def robR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        memo = defaultdict(lambda:-1)
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return nums[0]
        return max(self.robHelperR(memo, nums, 0, len(nums)-2), self.robHelperR(memo, nums, 1, len(nums)-1))
    
    def robHelperDP(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return 0
        elif len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums)
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(dp[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        return dp[len(nums)-1]
    
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return max(self.robHelperDP(nums[len(nums) != 1:]), self.robHelperDP(nums[:-1]))
    
    def isEqual(self, a, b):
        """
        :type a: char
        :type b: char
        :rtype: bool
        """
        if a == b or b == '.':
            return True
        return False

    def isMatchR(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if len(p) > 1 and len(s) > 0:
            isEqual = self.isEqual(s[0], p[0])
            if p[1] == '*':
                ignore = self.isMatch(s, p[2:])
                once =  isEqual and self.isMatch(s[1:], p[2:]) 
                mult = isEqual and self.isMatch(s[1:], p) 
                return ignore or once or mult
            elif isEqual:
                return self.isMatch(s[1:], p[1:])
            else:
                return False
        elif len(s) > len(p):
            return False
        elif len(p) > 0 and len(s) == 0:
            if len(p) >= 2 and p[1] == '*':
                return self.isMatch(s, p[2:])
            else:
                return False
        else:
            return self.isEqual(s, p)

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = [[False for x in range(len(s)+1)] for x in range(len(p)+1)]
        dp[0] = [True] + [False] * (len(s))
        for k in range(1, len(p) + 1):
            if p[k-1] == '*':
                dp[k][0] = dp[k-2][0]
        for i in range(1, len(p) + 1):
            for j in range(1, len(s) + 1):
                if self.isEqual(s[j-1], p[i-1]):
                    dp[i][j] = dp[i-1][j-1]
                elif p[i-1] == '*':
                    dp[i][j] = dp[i-2][j] or (self.isEqual(s[j-1], p[i-2]) and dp[i][j-1])
        return dp[len(p)][len(s)]