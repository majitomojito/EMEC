from django.db import models

# Create your models here.
class Article(models.Model):

    title = models.CharField(max_length=42, default='No title')
    url = models.CharField(max_length=42, null=True, blank=True)
    content = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    created_on = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to = 'media/')
    category = models.ForeignKey('Category', on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.title

class Category(models.Model):
    name = models.CharField(max_length=42)

    def __str__(self):
        return self.name