from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False, verbose_name='主键')),
                ('youjian', models.CharField(default='', max_length=255, verbose_name='短信内容')),
                ('lable', models.IntegerField(default=0, verbose_name='分类标签')),
            ],
            options={
                'verbose_name': '数据集',
                'verbose_name_plural': '数据集',
            },
        ),
        migrations.CreateModel(
            name='Stopword',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='', max_length=255, verbose_name='停用词')),
            ],
            options={
                'verbose_name': '停用词表',
                'verbose_name_plural': '停用词表',
            },
        ),
    ]
