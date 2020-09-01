from rest_framework import serializers
from .models import Bus_Station, Bus, Selected_List


class BusStationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bus_Station
        #fields =('bus_station_num', 'bus_list')
        fields = '__all__'

class BusSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bus
        fields = '__all__'

class SelectedListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Selected_List
        fields = '__all__'