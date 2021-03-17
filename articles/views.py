from django.shortcuts import render
from django.http.response import JsonResponse
# Create your views here.
from django.http import HttpResponse
from django.views import generic
from .models import *
from tensorflow import keras
from PIL import Image
from io import BytesIO
import base64
import re
import numpy as np
import json 

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
        canvas =  request.POST.get('canvas')
        


        canvas = re.sub('^data:image/.+;base64,', '', canvas)
        #print("canvas", canvas)
        im = Image.open(BytesIO(base64.b64decode(canvas)))
        im = im.resize((28,28))
        im = im.convert('L')
        print("image size", im.size)


        container = np.zeros((6,28,28), dtype='float64')
        im = np.asarray(im)

        print(im)
        img = Image.fromarray(im, 'L')

        im = ((im / 255.0) - 1 ) * -1
        im[im < 0] = 0
        container[0] = im.copy()

        #img.show()
        #print(container)

        #img = Image.fromarray(container[0]  * 255.0)
        #img.show()

        new_model = keras.models.load_model('path_to_my_model.h5')
        print(new_model)
        prediction = new_model.predict(container)

        print("prediction", prediction)
        return JsonResponse({'data': str(np.argmax(prediction)), 'prediction': json.dumps(prediction[0].tolist())})
