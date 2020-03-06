import pandas as pd
import trip_advisor_consts
import hotels
clusters = trip_advisor_consts.CLUSTERS


def get_attr_dict():
	destinations = pd.read_csv('preprocessing/destinations_with_static_info.csv')
	print(destinations)
	sample_dests = list(zip(list(destinations['city']),list(destinations['trip_advisor_id'])))
	attractions_dict = {}
	name_desc_img = {}

	for city,t_id in sample_dests:
		data = hotels.get_attractions(t_id)
		name_desc_img[city] = data[0]
		attractions_dict[city] = data[1]
	return [name_desc_img,attractions_dict]




				

	