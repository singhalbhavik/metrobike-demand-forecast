-- Hourly trip-start counts per station (the demand signal).
-- Groups only two cheap columns; no SELECT * avoids scanning wide rows.
SELECT
  start_station_id                       AS station_id,
  TIMESTAMP_TRUNC(start_time, HOUR)      AS hour_utc,
  COUNT(*)                               AS demand
FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
WHERE start_station_id IS NOT NULL
  AND start_time       IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
