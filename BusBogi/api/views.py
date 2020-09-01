from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Bus_Station, Bus, Selected_List
from .serializers import BusSerializer, BusStationSerializer, SelectedListSerializer
from rest_framework import status

class BusStationView(APIView):

    def post(self, request):
        #bus_serializer = BusSerializer(data=request.data)

        busStation_serializer = BusStationSerializer(data=request.data)  # Request의 data를 UserSerializer로 변환

        if busStation_serializer.is_valid():
            busStation_serializer.save()  # UserSerializer의 유효성 검사를 한 뒤 DB에 저장
            return Response(busStation_serializer.data, status=status.HTTP_201_CREATED)  # client에게 JSON response 전달
        else:
            return Response(busStation_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request,  **kwargs):
        if kwargs.get('bus_station_num') is None:
            busStation_queryset = Bus_Station.objects.all() #모든 User의 정보를 불러온다.
            busStation_queryset_serializer = BusStationSerializer(busStation_queryset, many=True)
            return Response(busStation_queryset_serializer.data, status=status.HTTP_200_OK)
        else:
            bus_station_id = kwargs.get('bus_station_num')
            busStation_serializer = BusStationSerializer(Bus_Station.objects.get(bus_station_num=bus_station_id)) #id에 해당하는 User의 정보를 불러온다
            return Response(busStation_serializer.data, status=status.HTTP_200_OK)

class SelectView(APIView):
    def post(self, request):
        selected_serializer = SelectedListSerializer(data=request.data)  # Request의 data를 UserSerializer로 변환

        if selected_serializer.is_valid():
            selected_serializer.save()
            return Response(selected_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(selected_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request,  **kwargs):
        if kwargs.get('bus_station_num') is None:
            select_queryset = Selected_List.objects.all() #모든 User의 정보를 불러온다.
            select_queryset_serializer = SelectedListSerializer(select_queryset, many=True)
            return Response(select_queryset_serializer.data, status=status.HTTP_200_OK)
        else:
            bus_station_id = kwargs.get('bus_station_num')
            busStation_serializer = BusStationSerializer(Bus_Station.objects.get(bus_station_num=bus_station_id)) #id에 해당하는 User의 정보를 불러온다
            return Response(busStation_serializer.data, status=status.HTTP_200_OK)