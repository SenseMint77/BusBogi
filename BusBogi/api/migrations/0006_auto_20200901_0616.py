# Generated by Django 3.1 on 2020-09-01 06:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_selected_list'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bus',
            name='bus_num',
            field=models.IntegerField(primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='bus_station',
            name='bus_station_num',
            field=models.IntegerField(primary_key=True, serialize=False, unique=True),
        ),
        migrations.DeleteModel(
            name='Selected_List',
        ),
    ]
