# k-nn的算法实现
import pandas as pd

row_data = {
    "电影名字" : ['无问西东', '后来的我们', '前任三' , '红海行动' , '唐人街探案' , '战狼二' ],
    "打斗镜头" : [1,5,12,108,112,115],
    "接吻镜头" : [101,89,97,5,9,8],
    "电影类型" : ['爱情片', "爱情片", '爱情片', "动作片" , "动作片" , "动作片"]
}

movie_data = pd.DataFrame(row_data)
print(movie_data.head(5))
test = movie_data.iloc[:,1:3]


print('----------test---------')
print(test.min())
print(test.max())
print('------------------------')



new_data = [24,67]

print(movie_data.iloc[:,1:3] - new_data )
print("-----------")
print( (((movie_data.iloc[:,1:3] - new_data) ** 2).sum(axis=1) ) ** 0.5)

dist = list((((movie_data.iloc[:,1:3] - new_data) ** 2).sum(axis=1) ) ** 0.5)
print(dist)

dist_label = pd.DataFrame({
    'dist':dist,
    'label': (movie_data.iloc[:,3])
})

print(dist_label)
#          dist label
# 0   41.048752   爱情片
# 1   29.068884   爱情片
# 2   32.310989   爱情片
# 3  104.403065   动作片
# 4  105.394497   动作片
# 5  108.452755   动作片

dr = dist_label.sort_values(by = 'dist')[:4]
print(dr)

re = dr.loc[:, 'label'].value_counts()

print(re.index[0])

result = []
result.append(re.index[0])
print(result)

# 0-1归一化

def minmax(dataset):
    minDF = dataset.min()
    maxDF = dataset.max()
    normSet = (dataset - minDF)/(maxDF - minDF)
    return normSet


#  切分数据集
def randSplit(dataset, rate=0.9):
    n = dataset.shape[0]
    m = int(n*rate)
    train = dataset.iloc[:m,:]
    test = dataset.iloc[m:,:]
    test.index = range(test.shape[0])
    return train,test

def datingClass(train, test,k):
    n = train.shape[1] - 1
    m = test.shape[0]
    result = []
    for i in range(m):
        dist = list(((( train.iloc[:,:n] - test.iloc[i,:n] ) ** 2 ).sum(axis=1)**0.5 ))
        dist_label = pd.DataFrame( {
            'dist':dist,
            'lable':(train.iloc[:, n] )
        } )
        dr = dist_label.sort_values(by='dist')[:k]
        re = dr.loc[:,'label'].value_counts()
        result.append(re.index[0])
    result = pd.Series(result)
    test['predict'] = result
    acc = (test.iloc[:,-1] == test.iloc[:,-1]).mean()
    return test



