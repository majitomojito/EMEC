from django.shortcuts import render
from django.http.response import JsonResponse
# Create your views here.
from django.http import HttpResponse
from django.views import generic
from .models import *

import numpy as np
import json
from .functions import simulate

def index(request):
    articles = Article.objects.all()
    categories = Category.objects.all()
    context = {
        'articles': articles,
        'categories': categories
        }
    return render(request, 'welcome.html', context)

class ArticleDetail(generic.DetailView):
    model = Article
    template_name = 'article_detail.html'

    def get_context_data(self, **kwargs):
        this_article = Article.objects.get(id=self.kwargs['pk'])
        context = super(ArticleDetail, self).get_context_data(**kwargs)

        return context

def predict_number(request, *args, **kwargs):
    
    if request.method == 'POST':
        print(request)

        df = simulate()
        df_dict = df.to_dict('index')
        result = []
        colors = []
        for key in df_dict:
            result.append(df_dict[key])
            if df_dict[key]['Infectious']:
                colors.append('red')
            else:
                colors.append('black')

        return JsonResponse({'data': result, 'colors': colors})
