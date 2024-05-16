from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from recognition import rec
from . import models


@login_required
def index(request):
    if request.method == 'GET':
        results = models.Dataset.objects.all()[:1000]
        Search = request.GET.get('Search', '')
        if Search:
            results = models.Dataset.objects.raw('select * from dataset where youjian= ' + Search)
            models.Dataset.objects.filter(youjian__icontains=Search)
        return render(request, 'SPAM/table.html', locals())


@login_required
def huizong(request):
    if request.method == 'GET':
        return render(request, 'SPAM/huizong.html', locals())
    elif request.method == 'POST':
        Search = request.POST.get('Search', '')
        predict = rec(Search)
        if predict == "0":
            result = '非垃圾短信'
        elif predict == "1":
            result = '垃圾短信'
        return render(request, 'SPAM/huizong.html', locals())


@login_required
def yuce(request):
    datas1 = models.Dataset.objects.filter(lable=0).count()
    datas2 = models.Dataset.objects.filter(lable=1).count()
    print(datas1, datas2)
    dict_value = [{"name": '垃圾短信', "value": datas1}, {"name": '非垃圾短信', "value": datas2}]
    return render(request, 'SPAM/yuce.html', locals())
