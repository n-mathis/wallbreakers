from collections import defaultdict
from collections import Counter

class TrieNode():
    
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = None
        
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        root = self.root
        for c in word:
            root = root.children[c]
        root.isWord = word

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        root = self.root
        for c in word:
            root = root.children.get(c)
            if not root:
                return False
        return root.isWord

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        root = self.root
        for c in prefix:
            root = root.children.get(c)
            if root == None:
                return False
        return True

    def helperLongestWord(self, word):
        root = self.root
        for c in word:
            root = root.children.get(c)
            if not root.isWord:
                return False
        return root.isWord

class MyHashSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.capacity = 10000
        self.size = 0
        self.set = [None] * self.capacity
    
    def hash(self, value):
        return value % self.capacity

    def add(self, value):
        """
        :type value: int
        :rtype: None
        """
        if self.contains(value):
            return
        hashKey = self.hash(value)
        if not self.set[hashKey]:
            self.set[hashKey] = [value]
            self.size += 1
        else:
            self.set[hashKey].append(value)
            self.size += 1
        
    def remove(self, value):
        """
        :type value: int
        :rtype: None
        """
        if not self.contains(value):
            return
        hashKey = self.hash(value)
        for v in self.set[hashKey]:
            if v == value:
                self.set[hashKey].remove(value)
                self.size -= 1

    def contains(self, value):
        """
        Returns true if this set contains the specified element
        :type value: int
        :rtype: bool
        """
        hashKey = self.hash(value)
        if self.set[hashKey] == None:
            return False
        for v in self.set[hashKey]:
            if v == value:
                return True
        return False

class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.buckets = 10000
        self.size = 0
        self.maps = [[] for x in range(self.buckets+1)]
        
    def hash(self, key):
        return key % self.buckets
    
    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        hashKey = self.hash(key)
        for i, pair in enumerate(self.maps[hashKey]):
            if pair[0] == key:
                self.maps[hashKey][i][1] = value
                return
        self.maps[hashKey].append([key, value])

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        hashKey = self.hash(key)
        for pair in self.maps[hashKey]:
            if pair[0] == key:
                return pair[1]
        return -1

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        hashKey = self.hash(key)
        bucket = self.maps[hashKey]
        for i, pair in enumerate(bucket):
            if pair[0] == key:
                if i < len(bucket)-1:
                    bucket[i] = bucket[-1]
                bucket.pop()
                return

class Solution(object):

    def findWordsHelper(self, board, i, j, node, found):
        """
        :type board: List[List[str]]
        :type node: TrieNode
        :type i: int
        :type j: int
        :type path: str
        :type found: List[str]
        """
        if node.isWord:
            found.append(node.isWord)
            node.isWord = None
            
        if i < len(board) and i >= 0 and j < len(board[0]) and j >= 0 and board[i][j] in node.children:
            temp = board[i][j]
            board[i][j] = '#'
            self.findWordsHelper(board, i+1, j, node.children[temp], found)
            self.findWordsHelper(board, i-1, j, node.children[temp], found)
            self.findWordsHelper(board, i, j-1, node.children[temp], found)
            self.findWordsHelper(board, i, j+1, node.children[temp], found)
            board[i][j] = temp
        
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        found = []
        trie = Trie()
        for word in words:
            trie.insert(word) 
        root = trie.root
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.findWordsHelper(board, i, j, root, found)
        return found

    def longestWord(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        trie = Trie()
        for word in words:
            trie.insert(word)
        longest = ""
        for word in words:
            if len(word) < len(longest) or len(word) == len(longest) and word > longest:
                continue
            if trie.helperLongestWord(word):
                longest = word
                
        return longest

    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        startIs = []
        pCounter = Counter(p)
        sCounter = Counter(s[:len(p)-1])
        for i in range(len(p)-1,len(s)):
            sCounter[s[i]] += 1  
            if sCounter == pCounter: 
                startIs.append(i-len(p)+1)   
            sCounter[s[i-len(p)+1]] -= 1  
            if sCounter[s[i-len(p)+1]] == 0:
                del sCounter[s[i-len(p)+1]]   
        return startIs
        
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        sCounter = Counter(s)
        for i in range(len(s)):
            if sCounter[s[i]] == 1:
                return i
        return -1

    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        sCounter = Counter(s)
        tCounter = Counter(t)
        return list(tCounter - sCounter)[0]
  
    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        for c in "!?',;.":
            paragraph = paragraph.replace(c, " ")
        ban = set(banned)
        words = paragraph.lower().split()
        pCounter = Counter(word for word in words if word not in ban).most_common(1)
        return pCounter[0][0]

    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        freq = ""
        sCounter = Counter(s).most_common(len(s))
        for i in sCounter:
            freq += i[0]*i[1]
        return freq
    
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        aimSum = sum(range(1, len(nums)+1))
        sumNumsSet = sum(set(nums))
        sumNums = sum(nums)
        missing = aimSum - sumNumsSet
        repeated = sumNums - sumNumsSet
        return [repeated, missing]

     def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        subCount = []
        counter = Counter()
        for cpd in cpdomains:
            n, d = cpd.split()
            counter[d] += int(n)
            for i in range(len(d)):
                if d[i] == '.':
                    counter[d[i + 1:]] += int(n)
        for i in counter:
            subCount.append(str(counter[i]) + " " + i)
        return subCount

    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return set(nums1).intersection(set(nums2))

    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        s = set()
        while n != 1:
            n = sum([int(i) ** 2 for i in str(n)])
            if n in s:
                return False
            s.add(n)
        return True

    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        total = sum([S.count(j) for j in J])
        return total
    
    def uncommonFromSentences(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: List[str]
        """
        orig = []
        c = Counter((A + " " + B).split())
        for w in c:
            if c[w] == 1:
                orig.append(w)
        return orig

    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        return min(len(candies) / 2, len(set(candies)))
    
    def numSpecialEquivGroups(self, A):
        """
        :type A: List[str]
        :rtype: int
        """
        B = ["".join(sorted(a[0::2])) + "".join(sorted(a[1::2])) for a in A]
        s = set(B)
        return len(s)
    
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        sD = {}
        tD = {}
        for i, val in enumerate(s):
            sD[val] = sD.get(val, []) + [i]
        for i, val in enumerate(t):
            tD[val] = tD.get(val, []) + [i]
        return sorted(sD.values()) == sorted(tD.values())

    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        pD = {}
        sD = {}
        for i, val in enumerate(pattern):
            pD[val] = pD.get(val, []) + [i]
        for i, val in enumerate(str.split(" ")):
            sD[val] = sD.get(val, []) + [i]
        return sorted(pD.values()) == sorted(sD.values())

    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        ws = []
        for word in words:
            w=''
            for i in word:
                l = morse[ord(i) - ord('a')]
                w += l
            ws.append(w)
        return len(set(ws))

    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        seen = []
        for i, row in enumerate(board):
            for j, n in enumerate(row):
                if n != '.':
                    seen += [(n,j),(i,n),(i/3,j/3,n)]
        return len(seen) == len(set(seen))

    def countOfAtoms(self, formula):
        """
        :type formula: str
        :rtype: str
        """
        reverse = formula[::-1]
        num = ''
        element = ""
        multipliers = [1]
        counts = Counter()
        for char in reverse:
            if char.isdigit():
                num = char + num
            elif char == ")":
                multipliers.append(int(num)*int(multipliers[-1]))
                num = ''         
            elif char == "(":
                multipliers.pop()
            elif char.isalpha() and char.islower():
                element = char+element
            elif char.isalpha() and char.isupper():
                element =char+element
                if num != '':
                    counts[element] += int(num)*int(multipliers[-1])
                else:
                    counts[element] +=1*int(multipliers[-1])
                element = '' 
                num = ''
        ret = []
        for key in sorted(counts.keys()):
            if counts[key] > 1:
                ret.append(key+str(counts[key])) 
            else:
                ret.append(key)
        return "".join(ret)