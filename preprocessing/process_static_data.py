import pandas as pd
import process_city_keywords
import country
import language
import safety
import get_trip_advisor_static

#df=process_city_keywords.df
df=pd.read_csv("/scraping/destinations.csv")

df['country']=df.apply(lambda row: country.get_country(row['city']),axis=1)
df['language']=df.apply(lambda row: language.get_language(row['country']),axis=1)
df['safety']=df.apply(lambda row: safety.get_safety(row['country']), axis=1)

df['trip_advisor_id']=df.apply(lambda row: get_trip_advisor_static.get_location_id(row['city']))
df['attractions']=df.apply(lambda row: get_trip_advisor_static.get_location_id(row['trip_advisor_id']))


df.to_csv("/CMSC-12200-Group-Project/destinations_with_static_info.csv", header=True, index=False)