from django.contrib import admin
from .models import NewsDataModel, PredictionModel

# Register your models here.
admin.site.register(NewsDataModel)
admin.site.register(PredictionModel)
