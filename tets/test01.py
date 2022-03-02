class Solution:
    def minimumFinishTime(self, tires: List[List[int]], changeTime: int, numLaps: int) -> int:
        def cishu(f, i):
            n = 0
            ret = 0
            while ret < changeTime:
                ret += f * (i ** n)
                n += 1
            return n

        tires = sorted(tires,key=lambda x:cishu(x[0],x[1]), reverse=True)
        
        print(tires)


