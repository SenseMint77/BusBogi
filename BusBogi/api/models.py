from django.db import models
from django_mysql.models import ListCharField

# Create your models here.

class Bus(models.Model):
    #id = models.AutoField()
    bus_num = models.IntegerField(null= False, primary_key= True, unique= True)

    class Meta:
        db_table = "Bus"

class Bus_Station(models.Model):
    #id = models.AutoField()
    bus_station_num = models.IntegerField(null= False, primary_key= True, unique=True)
    bus_station_name = models.CharField(max_length= 30, null= True)
    bus_list = models.ManyToManyField(Bus)
    #bus_list1 = models.ForeignKey()

    class Meta:
        db_table = "Bus_Station"

class Selected_List(models.Model):
    user_id = models.CharField(max_length= 12, null= False, primary_key= True, unique= True)
    selected_list = models.ManyToManyField(Bus)
    #selected_list = ListCharField(max_length= 12,base_field=models.IntegerField())

    class Meta:
        db_table = "Selected_List"


# bus1 = Bus.objects.create(bus_num = 101)
# bus2 = Bus.objects.create(bus_num = 11)
# bus3 = Bus.objects.create(bus_num = 12)
# bus4 = Bus.objects.create(bus_num = 13)
# bus5 = Bus.objects.create(bus_num = 14)
#
#
# busList1 = Bus_Station.objects.create(bus_station_num = 10112, bus_station_name = "station1")
# busList2 = Bus_Station.objects.create(bus_station_num = 10101, bus_station_name = "station2")
# busList3 = Bus_Station.objects.create(bus_station_num = 10108, bus_station_name = "station3")
# busList4 = Bus_Station.objects.create(bus_station_num = 10002, bus_station_name = "station4")
# busList5 = Bus_Station.objects.create(bus_station_num = 10672, bus_station_name = "station5")
#
#
# busList1.bus_list.add(bus1.bus_num, bus5.bus_num, bus4.bus_num)
# busList2.bus_list.add(bus2.bus_num, bus5.bus_num, bus3.bus_num)
# busList3.bus_list.add(bus4.bus_num, bus1.bus_num, bus5.bus_num)
# busList4.bus_list.add(bus3.bus_num, bus4.bus_num, bus2.bus_num)
# busList5.bus_list.add(bus5.bus_num, bus1.bus_num, bus4.bus_num)
