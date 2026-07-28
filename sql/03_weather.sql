-- Daily Austin weather from NOAA GSOD.
-- Austin-Bergstrom International Airport: USAF 722530.
-- Wildcard table suffix limits the scan to needed years.
-- 9999.9 / 99.99 are NOAA sentinel values for missing observations.
SELECT
  PARSE_DATE('%Y%m%d', CONCAT(year, mo, da))              AS date,
  IF(temp >= 9999.0, NULL, CAST(temp  AS FLOAT64))        AS temp_f,
  IF(prcp >= 99.9,   NULL, CAST(prcp  AS FLOAT64))        AS precip_in
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE stn = '722530'
  AND _TABLE_SUFFIX BETWEEN '{start_year}' AND '{end_year}'
ORDER BY date
