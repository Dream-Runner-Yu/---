
-- 历史热门电影
select mid, count(mid) as count from rating group by mid

-- 近期热门统计
with ratingOfMonth as
select mid,
       score,
       changDate(timestamp) as  yearmonth
from ratings

select mid, yearmonth,
       count(mid) as count
from ratingOfMonth
group by
        yearmonth,
        mid
order by yearmonth desc, count desc ;

-- 电影平均评分统计
select mid, avg(score) as avg from ratings group by mid;

-- 各类别的 Top10
with movieWithScore as
    select a.mid,
           a.genres,
           if(isnull(b.avg),0,b.avg) score
    from movies a
    left join averageMovies b
    on a.mid = b.mid

select mid,
       gen,
       score,
       row_number() over(partition by gen order by scoe desc) rank
from (
         select mid, score, explode(splitGe(genres)) gen from movieWithScore
         from ;
     )
-- where rank <= 10