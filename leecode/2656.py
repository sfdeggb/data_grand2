class Solution(object):
    """
    给你一个下标从 0 开始的整数数组 nums 和一个整数 k 。
    你可以对数组执行 最多 k 次操作。在一次操作中，你可以选择数组中的任一元素，并增加它。
    请你返回数组在执行 最多 k 次操作后可以得到的最大和。
    """
    def maximizeSum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        slected_element=[]
        for num in range(k):
            max_num = max(nums)
            slected_element.append(max_num)
            nums.remove(max_num)
            nums.append(max_num+1)
        return sum(slected_element)

if __name__ == "__main__":
    nums = [1,2,3,4,5]
    k = 3
    print(Solution().maximizeSum(nums, k))