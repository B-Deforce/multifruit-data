SELECT 
    * 
FROM 
    vol_water_content
WHERE 
    vmc_10 != -1
    AND vmc_30 != -1
    AND vmc_50 != -1
    AND vmc_70 != -1;