# Generated by Django 2.2.5 on 2020-03-09 16:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('question', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='question',
            name='test',
        ),
        migrations.AddField(
            model_name='question',
            name='title',
            field=models.CharField(default='Question 1', max_length=10),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='question',
            name='answer',
            field=models.CharField(choices=[('Y', 'Yes'), ('N', 'No'), ('?', 'Not Sure')], max_length=1),
        ),
    ]
