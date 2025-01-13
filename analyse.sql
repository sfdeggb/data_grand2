<<<<<<< HEAD
-- 选择热门视频
select video_id, count(*) as watch_count
from watch_history
group by video_id
order by watch_count desc
limit 10;
-- 热门视频类目分布
select category_id, count(*) as watch_count
from watch_history
group by category_id
order by watch_count desc
limit 10;

--热门视频5s播放占比
select video_id, count(*) as watch_count
from watch_history
where watch_time >= 5
group by video_id
order by watch_count desc
limit 10;

-- 选择质量视频（5s播放占比）
select video_id, count(*) as watch_count
from watch_history
where watch_time >= 5
group by video_id
order by watch_count desc
limit 10;

-- 质量视频类目分布
select category_id, count(*) as watch_count
from watch_history
where video_id in (select video_id from quality_video)
group by category_id
order by watch_count desc
limit 10;

-- 曝光视频5s播放占比
select video_id, count(*) as watch_count
from watch_history
where watch_time >= 5
group by video_id
order by watch_count desc
limit 10;

-- 曝光视频/营销视频占比
select video_id, count(*) as watch_count
from watch_history
where video_id in (select video_id from exposure_video)
group by video_id
order by watch_count desc
limit 10;

--人均观看类目数
select category_id, count(*) as watch_count
from watch_history
group by category_id
order by watch_count desc
limit 10;

=======
-- 现在有四个表分别是：短视频属性表 曝光表 用户画像表  
-- 热门视频（视频播放次数排序）
>>>>>>> 1365ffebaf30cdb251c7dd41657efa08336b35ae
