from django.urls import path
from . import views

app_name = 'api'
urlpatterns = [
    path('station/bus/', views.BusStationView.as_view()),
    path('station/bus/<int:bus_station_num>', views.BusStationView.as_view()),
    path('station/bus/select', views.SelectView.as_view())
]