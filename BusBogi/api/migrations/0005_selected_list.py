# Generated by Django 3.1 on 2020-08-31 02:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_auto_20200827_0908'),
    ]

    operations = [
        migrations.CreateModel(
            name='Selected_List',
            fields=[
                ('user_id', models.CharField(max_length=12, primary_key=True, serialize=False, unique=True)),
                ('selected_list', models.ManyToManyField(to='api.Bus_Station')),
            ],
            options={
                'db_table': 'Selected_List',
            },
        ),
    ]
