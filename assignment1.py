import math

class Solution(object):

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dic = {}
        for i, num in enumerate(nums):
            remainder = target - num
            if remainder in dic:
                return dic[remainder],i
            dic[num] = i

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        prefix = ''
        if strs == []:
            return prefix
        else:
            i = 0
            while(True):
                currL = 0
                for word in strs:
                    if i == len(word):
                        return prefix
                    if currL == 0:
                        currL = word[i]
                    if currL != word[i]:
                        return prefix
                prefix += currL
                i += 1

    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        str1 = ''.join(str(e) for e in digits)
        
        integer = int(str1)+1
        return map(int, str(integer))

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = (''.join([i for i in s if i.isalpha() or i.isdigit()])).lower()
        j = len(s)-1
        for i in range(len(s)/2):
            if s[i] != s[j]:
                return False
            j -= 1
        return True

    def dfsSolve(self,board, x, y):
        """
        :type board: List[List[str]]
        :type x: int
        :type y: int
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if x < 0 or x > len(board)-1 or y < 0 or y > len(board[0])-1 or board[x][y]!='O':
            return
        board[x][y] = 'V'
        self.dfs(board,x-1, y)
        self.dfs(board,x+1, y)
        self.dfs(board,x, y+1)
        self.dfs(board,x, y-1)

    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if len(board) == 0:
            return
        rows = len(board)
        cols = len(board[0])
        for i in range(rows):
            self.dfs(board,i, 0)
            self.dfs(board,i, cols-1)
        for j in range(1, cols-1):
            self.dfs(board,0, j)
            self.dfs(board,rows-1, j)
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'V':
                    board[i][j] = 'O'
        
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        num = 0
        for i in range(len(nums)):
            num = num ^ nums[i]
        return num

    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.upper()
        LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        maps = {}
        for i in range(26):
            maps[LETTERS[i]] = i + 1
        col = 0 
        for l in s:
            col = 26*col + maps[l]
        
        return col

    # helper for numIslands using dfs
    def dfsNumIslands(self, grid, x, y):
        """
        :type grid: List[List[str]]
        :type x: int
        :type y:
        :rtype: int
        """
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[x]) or grid[x][y] == '0':
            return 0
        grid[x][y] = '0'
        self.dfsNumIslands(grid, x+1, y)
        self.dfsNumIslands(grid, x-1, y)
        self.dfsNumIslands(grid, x, y+1)
        self.dfsNumIslands(grid, x, y-1)
        return 1
        
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if grid == [[]]:
            return 0
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '1':
                    count += self.dfsNumIslands(grid, i, j)
        return count

    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if (n == 0): 
            return False
        while (n != 1): 
            if (n % 2 != 0): 
                return False
            n = n // 2
        return True

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        else:
            s = sorted(s)
            t = sorted(t)
            for i in range(len(s)):
                if s[i] != t[i]:
                    return False
            return True

    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        for i in range(len(s)//2):
            s[i],s[len(s)-1-i] = s[len(s)-1-i],s[i]
        return s

    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        indices = []
        vowels = ''
        for i in range(len(s)):
            if s[i] in 'aeiouAEIOU':
                vowels += s[i]
                indices.append(i)
        vowels = vowels[::-1]
        reverseS = list(s)
        for i in range(len(indices)):
            reverseS[indices[i]] = vowels[i] 
        return ''.join(reverseS)

    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        fb = []
        for i in range(1,n+1):
            if i%3 == 0:
                if i%5==0:
                    fb+=["FizzBuzz"]
                else:
                    fb+=["Fizz"]
            elif i%5==0:
                fb+=["Buzz"]
            else:
                fb+=[str(i)]
        return fb
            
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        diff = bin(x ^ y)
        count = diff.count('1')
        return count

    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        complement = ''
        bNum = bin(num)[2:]
        for i in range(len(bNum)):
            if bNum[i] == '0':
                complement += '1'
            elif bNum[i] == '1':
                complement += '0'
        return int(complement, 2)

    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        if word.isupper():
            return True
        elif word.islower():
            return True
        elif word[0].isupper() and word[1:].islower():
            return True
        else:
            return False
    
    # helper for findCircleNum using dfs
    def dfsfindCircleNum(self, M, x, visited):
        """
        :type M: List[List[str]]
        :type x: int
        :type visited: set
        :rtype: int
        """
        for y, isFriend in enumerate(M[x]):
            if isFriend and y not in visited:
                visited.add(y)
                self.dfsfindCircleNum(M, y, visited)
        
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        count = 0
        visited = set()
        for r in range(len(M)):
            if r not in visited:
                count += 1
                self.dfsfindCircleNum(M, r, visited)
        return count
    
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        listS = s.split()
        reverseWords = []
        for i in listS:
            reverseWords +=  [i[::-1]]
        
        strRW = ' '.join(reverseWords)
        return(strRW)

    #helper for selfDividingNumbers
    def isSelfDivide(self, n):
        """
        :type n: int
        :rtype: bool
        """
        string = str(n)
        for j in string:
            if int(j)==0 or n%int(j)!= 0:
                return False
        return True
    
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        l = range(left,right+1)
        out = []
        for i in range(len(l)):
            if self.isSelfDivide(l[i]):
                out += [l[i]]
        return out

    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        for row in A: 
            for i in range((len(row) + 1) // 2): 
                # swap and invert
                row[i],row[len(row) - 1 - i] = row[len(row) - 1 - i] ^ 1, row[i] ^ 1
  
        return A 

    def transpose(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        trans= [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))] 

        return trans

    def binaryGap(self, N):
        """
        :type N: int
        :rtype: int
        """
        b = bin(N)[2:]
        index = 0
        maxDis = 0
        for i, d in enumerate(b):
            if d == "1":
                maxDis = max(maxDis, i - index)
                index = i
        return maxDis

    def sortArrayByParity(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        evens = []
        odds = []
        for i in A:
            if i % 2 == 0:
                evens.append(i)
            else:
                odds.append(i)
        evens += odds
        return evens
