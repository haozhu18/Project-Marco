# Generated by Django 2.2.5 on 2020-03-10 02:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('question', '0004_auto_20200309_1644'),
    ]

    operations = [
        migrations.AlterField(
            model_name='question',
            name='answer',
            field=models.IntegerField(choices=[(0, 'Not Interested'), (1, 'A little Interested'), (2, 'Somewhat Interested'), (3, 'Very Interested'), (4, 'Really Interested'), (5, 'Really Very Interested')], max_length=1),
        ),
    ]
