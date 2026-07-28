-- Schema exploration: date range, station count, total trip volume.
-- Scans the full table once; used only during pipeline startup.
SELECT
  MIN(start_time)                   AS earliest_trip,
  MAX(start_time)                   AS latest_trip,
  COUNT(DISTINCT start_station_id)  AS station_count,
  COUNT(*)                          AS total_trips
FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
