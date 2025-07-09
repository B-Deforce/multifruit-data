WITH 
soil_moist_raw AS (
    SELECT * FROM soil_moisture 
    INNER JOIN field_sensor ON field_sensor.sensor_id=soil_moisture.sensor_id 
    INNER JOIN plot ON plot.plot_id=field_sensor.plot_id
),
soil_moist_grouped AS (
    SELECT 
        orchard_id, 
        plot_id, 
		measurement_year,
		measurement_date,
		depth_cm,
		AVG(soil_temp) as avg_soil_temp,
        AVG(soil_moisture) as avg_moist
    FROM soil_moist_raw
    GROUP BY orchard_id, plot_id, depth_cm, measurement_date
),
swp_raw AS (
    SELECT DATE(measurement_date) as measurement_date, * FROM swp 
    INNER JOIN plot ON plot.plot_id=swp.plot_id
),
ETo_raw AS (
    SELECT DATE(measurement_date) as measurement_date, * FROM ETo 
    INNER JOIN weather_sensor ON weather_sensor.sensor_id=ETo.sensor_id
),
orchard_raw AS (
    SELECT * FROM orchard
),
rainfall_raw AS (
    SELECT DATE(measurement_date) as measurement_date, * FROM precipitation_daily 
    INNER JOIN weather_sensor ON weather_sensor.sensor_id=precipitation_daily.sensor_id 
    INNER JOIN weather_sensor_alloc ON weather_sensor_alloc.sensor_id=weather_sensor.sensor_id
),
irrigation_raw AS (
    SELECT DATE(irrigation_date) as irrigation_date, * FROM irrigation_intervention 
    INNER JOIN plot ON plot.plot_id=irrigation_intervention.plot_id
),
treatment_raw AS (
    SELECT * FROM treatment 
    INNER JOIN plot ON plot.plot_id=treatment.plot_id
),
full_data AS (
    SELECT 
        soil_moist_grouped.*, 
        orchard_raw.orchard_name, 
        orchard_raw.soil_text_0to30cm, 
        ETo_raw.ETo, 
        rainfall_raw.precip_daily, 
        treatment_raw.irrigation_treatment,
		treatment_raw.pruning_treatment,
        irrigation_raw.irrigation_amount,
        swp_raw.swp_mpa
    FROM soil_moist_grouped 
    INNER JOIN orchard_raw ON orchard_raw.orchard_id = soil_moist_grouped.orchard_id
	INNER JOIN ETo_raw ON ETo_raw.measurement_date = soil_moist_grouped.measurement_date
	LEFT JOIN rainfall_raw ON rainfall_raw.measurement_date = soil_moist_grouped.measurement_date AND rainfall_raw.orchard_id = soil_moist_grouped.orchard_id
	LEFT JOIN treatment_raw ON treatment_raw.plot_id = soil_moist_grouped.plot_id AND treatment_raw.treatment_year = soil_moist_grouped.measurement_year AND treatment_raw.orchard_id = soil_moist_grouped.orchard_id
	LEFT JOIN irrigation_raw ON irrigation_raw.plot_id = soil_moist_grouped.plot_id AND irrigation_raw.irrigation_date = soil_moist_grouped.measurement_date AND irrigation_raw.orchard_id = soil_moist_grouped.orchard_id
	LEFT JOIN swp_raw ON swp_raw.measurement_date = soil_moist_grouped.measurement_date AND swp_raw.plot_id = soil_moist_grouped.plot_id AND swp_raw.orchard_id = soil_moist_grouped.orchard_id
	)
SELECT * FROM full_data;

