# Generated by Django 3.1 on 2020-08-27 09:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_auto_20200827_0905'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bus_station',
            name='bus_station_num',
            field=models.IntegerField(max_length=10, primary_key=True, serialize=False, unique=True),
        ),
    ]
